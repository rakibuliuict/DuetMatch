# ============================================================
# DuetMatch (your code) + Uncertainty-Aware Duet Loss
# + Uncertainty-Aware Pseudo-Label Refinement (CutMix/CPS)
#
# Notes:
# - I kept your training pipeline structure unchanged.
# - I ONLY replaced (1) the unlabeled duet losses and
#   (2) the CPS filtering masks with uncertainty-aware confidence.
# - I also fixed get_cut_mask to be 2D/3D safe (no dim mismatch).
# - Added robust shape helpers to avoid size mismatch errors.
# ============================================================

import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np

from skimage.measure import label
from medpy import metric

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.data import DataLoader

from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.brats19 import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing


# -----------------------------
# Argparse (unchanged)
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/Research/Dataset/brats2019', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='test', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=1000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=6000, help='maximum self-train iteration to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=10, help='trained samples')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default=10.0, help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()


# -----------------------------
# Dataset detect (unchanged)
# -----------------------------
dataset = 'test'
if 'brats2019' in args.root_path:
    dataset = 'brats2019'
elif 'brats2018' in args.root_path:
    dataset = 'brats2018'
elif 'brats2017' in args.root_path:
    dataset = 'brats2017'


# -----------------------------
# Global config (unchanged)
# -----------------------------
train_data_path = args.root_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (96, 96, 96)
num_classes = 2


# ============================================================
# Utility: safe masks, entropy, uncertainty weighting
# ============================================================

def ensure_same_ndim_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Make x have same ndim as ref by adding singleton dims at dim=1 end.
    Useful when context_mask returns [B,1,1,1,1] but you need [B,D,H,W] etc.
    """
    while x.ndim < ref.ndim:
        x = x.unsqueeze(1)
    while x.ndim > ref.ndim:
        x = x.squeeze(1)
    return x

def ensure_same_shape(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Broadcast x to ref shape if possible (by singleton expansion).
    """
    x = ensure_same_ndim_like(x, ref)
    # if still mismatch, try expand on singleton dims
    if x.shape != ref.shape:
        # Expand only where singleton
        expand_shape = []
        for a, b in zip(x.shape, ref.shape):
            if a == b:
                expand_shape.append(a)
            elif a == 1:
                expand_shape.append(b)
            else:
                # cannot expand safely
                return x
        x = x.expand(*expand_shape)
    return x

def soft_ce_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Soft cross entropy per-voxel:
      L(v) = - sum_c teacher_probs(c,v) * log softmax(student_logits)(c,v)
    Returns: [B, ...] (channel reduced)
    """
    logp = F.log_softmax(student_logits, dim=1)
    loss = -(teacher_probs * logp).sum(dim=1)  # [B, ...]
    return loss

def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    probs: [B, C, ...]
    returns entropy: [B, 1, ...]
    """
    H = -(probs * (probs + eps).log()).sum(dim=1, keepdim=True)
    return H

def beta_schedule(iter_num: int, max_iter: int,
                  beta_max: float = 1.0, beta_min: float = 0.1, decay: float = 0.1) -> float:
    """
    DyCON-style schedule:
      beta = max(beta_min, beta_max * exp(-decay * iter/max_iter))
    """
    t = float(iter_num)
    T = float(max_iter) + 1e-8
    beta = beta_max * np.exp(-decay * (t / T))
    beta = max(beta_min, beta)
    return float(beta)

def uncertainty_weight(Hs: torch.Tensor, Ht: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Hs, Ht: [B,1,...]
    beta: scalar tensor
    w: [B,1,...]
    """
    return 1.0 / (torch.exp(beta * Hs) + torch.exp(beta * Ht) + 1e-8)

def weighted_feature_mse(Fs: torch.Tensor, Ft: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Fs, Ft: [B, Cf, ...]
    w:      [B, 1,  ...]
    Feature MSE map -> mean over feature channels -> [B,...]
    """
    mse_map = (Fs - Ft).pow(2).mean(dim=1)  # [B,...]
    w2 = w.squeeze(1)                       # [B,...]
    w2 = ensure_same_shape(w2, mse_map)
    return (mse_map * w2).mean()

def confidence_from_entropy(probs: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    probs: [B,C,...]
    return conf: [B,1,...]
    """
    H = entropy_from_probs(probs)  # [B,1,...]
    conf = torch.exp(-kappa * H)
    return conf


# ============================================================
# Your helper funcs (mostly unchanged)
# ============================================================

def get_cut_mask(out, thres=0.5, nms=0):
    """
    FIXED: works for both 2D and 3D shapes.
    out: logits [B, C, ...]
    returns: mask [B, ...] for class 1
    """
    probs = F.softmax(out, dim=1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1].contiguous()  # [B, ...]
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    """
    segmentation: [B, ...] int/float
    returns: [B, ...] float cuda
    """
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels_ = label(n_prob)
        if labels_.max() != 0:
            largestCC = labels_ == (np.argmax(np.bincount(labels_.flat)[1:]) + 1)
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def labeled_ratio_to_patients(dataset_path, patiens_num):
    ref_dict = None
    if "brats2019" in dataset_path:
        ref_dict = {"4": 10, "10": 25, "20": 50}
    elif "brats2018" in dataset_path:
        ref_dict = {"10": 20}
    elif "brats2017" in dataset_path:
        ref_dict = {"10": 20}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def feature_l2_loss(student_feat, teacher_feat):
    return F.mse_loss(student_feat, teacher_feat)

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


# ============================================================
# Training: Pre-train (unchanged)
# ============================================================

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = BraTS(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ])
    )
    logging.info(f'Max samples: {len(db_train)}')

    labelnum = labeled_ratio_to_patients(args.root_path, args.labelnum)
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs].cuda()
            label_batch = sampled_batch['label'][:args.labeled_bs].cuda()

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2.0

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'
                         % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(
                    args.root_path, model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                model.train()
                logging.info("iteration %d, dice: %.04f" % (iter_num, dice_sample))

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break


# ============================================================
# Mix helper (unchanged)
# ============================================================

def mix_consecutive_pairs(tensor, mask):
    """
    tensor: [B, ...]
    mask:   broadcastable to [B, ...]
    """
    B = tensor.size(0)
    tensor2 = tensor.view(B // 2, 2, *tensor.shape[1:])
    mask2 = mask.view(B // 2, 2, *mask.shape[1:]) if mask.ndim == tensor.ndim else mask
    mixed = tensor2 * mask2 + tensor2.flip(1) * (1 - mask2)
    return mixed.view(B, *tensor.shape[1:])


# ============================================================
# Self-train: UPDATED with Uncertainty-Aware Duet Loss + UA PL refine
# ============================================================

def self_train(args, pre_snapshot_path, self_snapshot_path):
    encoder_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    decoder_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    inference_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(encoder_model, pretrained_model)
    load_net(decoder_model, pretrained_model)

    # Freeze decoder of encoder model
    for param in encoder_model.decoder.parameters():
        param.requires_grad = False
    # Freeze encoder of decoder model
    for param in decoder_model.encoder.parameters():
        param.requires_grad = False

    db_train = BraTS(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ])
    )

    labelnum = labeled_ratio_to_patients(args.root_path, args.labelnum)
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        args.batch_size, args.batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    DICE = losses.mask_DiceLoss(nclass=2)
    CE = nn.CrossEntropyLoss(reduction='none')

    encoder_optimizer = optim.SGD(encoder_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    decoder_optimizer = optim.SGD(decoder_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    encoder_model.train()
    decoder_model.train()

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0
    best_dice_1 = 0
    best_dice_2 = 0

    max_epoch = self_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    # Hyperparams for uncertainty losses
    alpha_feat = 0.5   # feature consistency weight
    mu_ent = 0.1       # entropy regularizer weight
    kappa_conf = 1.0   # confidence map sharpness

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            img = volume_batch[:args.labeled_bs]
            lab = label_batch[:args.labeled_bs]
            unimg = volume_batch[args.labeled_bs:]

            # -----------------------------
            # 1) Supervised losses (same)
            # -----------------------------
            lb_output_encoder_model, _ = encoder_model(img)
            lb_output_decoder_model, _ = decoder_model(img)

            lb_loss_encoder_model = 0.5 * F.cross_entropy(lb_output_encoder_model, lab) + 0.5 * DICE(lb_output_encoder_model, lab)
            lb_loss_decoder_model = 0.5 * F.cross_entropy(lb_output_decoder_model, lab) + 0.5 * DICE(lb_output_decoder_model, lab)

            # -----------------------------
            # 2) Uncertainty-Aware Duet Loss (replaces your unlabeled feature + unlabeled CE)
            # Teacher = encoder_model prediction/feature (stable)
            # Student = decoder_model prediction/feature with perturb
            # -----------------------------
            with torch.no_grad():
                ulb_logits_teacher, ulb_feat_teacher = encoder_model(unimg)   # [B,C,...], [B,F,...]
                p_t = F.softmax(ulb_logits_teacher, dim=1)

            ulb_logits_student, ulb_feat_student = decoder_model(unimg, feature_perturb='dropout', p=0.6)
            p_s = F.softmax(ulb_logits_student, dim=1)

            H_t = entropy_from_probs(p_t)   # [B,1,...]
            H_s = entropy_from_probs(p_s)

            beta_val = beta_schedule(iter_num, self_max_iterations, beta_max=1.0, beta_min=0.1, decay=0.1)
            beta_t = torch.tensor(beta_val, device=unimg.device, dtype=torch.float32)

            w = uncertainty_weight(H_s, H_t, beta_t)  # [B,1,...]

            # Feature consistency (uncertainty-weighted)
            ulb_loss_F = weighted_feature_mse(ulb_feat_student, ulb_feat_teacher.detach(), w)

            # Prediction consistency (uncertainty-weighted soft CE)
            pred_map = soft_ce_loss(ulb_logits_student, p_t.detach())  # [B,...]
            w_pred = w.squeeze(1)
            w_pred = ensure_same_shape(w_pred, pred_map)
            ulb_loss_P = (pred_map * w_pred).mean()

            # Entropy regularization
            ulb_loss_H = (H_s.mean() + H_t.mean())

            ulb_loss_u_duet = alpha_feat * ulb_loss_F + ulb_loss_P + mu_ent * beta_t * ulb_loss_H

            # Use the same unlabeled regularizer for both sub-optimizations (stable training)
            loss_encoder = lb_loss_encoder_model + 0.5 * ulb_loss_u_duet
            loss_decoder = lb_loss_decoder_model + 0.5 * ulb_loss_u_duet

            # -----------------------------
            # 3) Build inference_model (same as yours)
            # -----------------------------
            with torch.no_grad():
                inference_model.decoder.load_state_dict(encoder_model.decoder.state_dict())
                inference_model.encoder.load_state_dict(decoder_model.encoder.state_dict())

                ulb_output_encoder_model, _ = encoder_model(unimg)
                ulb_output_decoder_model, _ = decoder_model(unimg)
                ulb_output_inference_model, _ = inference_model(unimg)

                # pseudo-labels (hard) for mixing (same as yours)
                plab_ulb_encoder_model = get_cut_mask(ulb_output_encoder_model, nms=1)   # [B,...]
                plab_ulb_decoder_model = get_cut_mask(ulb_output_decoder_model, nms=1)   # [B,...]
                plab_ulb_inference_model = get_cut_mask(ulb_output_inference_model, nms=1)

                # context mask (same)
                img_mask, loss_mask = context_mask(unimg, args.mask_ratio)

                # -----------------------------
                # Uncertainty-Aware Pseudo-Label Refinement (CutMix/CPS):
                # Replace equality filtering with confidence from stable path entropy.
                # stable path = inference_model output probs
                # -----------------------------
                p_cons = F.softmax(ulb_output_inference_model, dim=1)           # [B,C,...]
                conf_map = confidence_from_entropy(p_cons, kappa=kappa_conf)    # [B,1,...]
                conf_map = conf_map.squeeze(1)                                  # [B,...]

            # -----------------------------
            # 4) CPS / CutMix mixing (same structure, only masks differ)
            # -----------------------------
            mixed_img = mix_consecutive_pairs(unimg, img_mask)

            # mix pseudo labels as before
            plab_mixed_encoder_model = mix_consecutive_pairs(plab_ulb_encoder_model, img_mask)
            plab_mixed_decoder_model = mix_consecutive_pairs(plab_ulb_decoder_model, img_mask)

            # OLD:
            # filtered_mask_encoder = (plab_ulb_encoder_model == plab_ulb_inference_model).float()
            # filtered_mask_decoder = (plab_ulb_decoder_model == plab_ulb_inference_model).float()
            #
            # NEW (uncertainty-aware):
            filtered_mask_encoder = conf_map
            filtered_mask_decoder = conf_map

            # Ensure loss_mask has same shape as filtered masks
            # loss_mask from context_mask sometimes is [B,1,...] or [B,...]
            loss_mask_ = loss_mask
            if isinstance(loss_mask_, torch.Tensor):
                if loss_mask_.ndim > filtered_mask_encoder.ndim:
                    loss_mask_ = loss_mask_.squeeze(1)
                loss_mask_ = ensure_same_shape(loss_mask_, filtered_mask_encoder)
            else:
                # fallback (should not happen)
                loss_mask_ = torch.ones_like(filtered_mask_encoder)

            # Your mixing logic (kept, but made shape-safe)
            mixed_mask_encoder = filtered_mask_encoder * loss_mask_ + filtered_mask_encoder.flip(0) * (1 - loss_mask_)
            mixed_mask_decoder = filtered_mask_decoder * loss_mask_ + filtered_mask_decoder.flip(0) * (1 - loss_mask_)

            mixed_output_encoder_model, _ = encoder_model(mixed_img)
            mixed_output_decoder_model, _ = decoder_model(mixed_img)

            # CPS losses (unchanged form)
            ce_enc = CE(mixed_output_encoder_model, plab_mixed_decoder_model.type(torch.int64))  # [B,...]
            ce_dec = CE(mixed_output_decoder_model, plab_mixed_encoder_model.type(torch.int64))  # [B,...]

            # Make masks shape-safe for CE maps
            mixed_mask_decoder_ = ensure_same_shape(mixed_mask_decoder, ce_enc)
            mixed_mask_encoder_ = ensure_same_shape(mixed_mask_encoder, ce_dec)

            loss_cps_encoder = (ce_enc * mixed_mask_decoder_).sum() / (mixed_mask_decoder_.sum() + 1e-16)
            loss_cps_decoder = (ce_dec * mixed_mask_encoder_).sum() / (mixed_mask_encoder_.sum() + 1e-16)
            loss_cps = loss_cps_encoder + loss_cps_decoder

            # -----------------------------
            # 5) Optimize (same)
            # -----------------------------
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = loss_encoder + loss_decoder + 0.5 * loss_cps
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            iter_num += 1

            logging.info('iteration %d : loss encoder: %03f, loss decoder: %03f, ulb_u_duet: %03f, cps: %03f'
                         % (iter_num, loss_encoder, loss_decoder, ulb_loss_u_duet, loss_cps))

            # EMA swap (kept same as yours)
            update_ema_variables(encoder_model.encoder, decoder_model.encoder, 0.99)
            update_ema_variables(decoder_model.decoder, encoder_model.decoder, 0.99)

            # -----------------------------
            # 6) Validation / saving (unchanged)
            # -----------------------------
            if iter_num % 200 == 0:
                encoder_model.eval()
                dice_sample_1 = test_3d_patch.var_all_case_LA(
                    args.root_path, encoder_model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if dice_sample_1 > best_dice_1:
                    best_dice_1 = round(dice_sample_1, 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_model1_dice_{}.pth'.format(iter_num, best_dice_1))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model1.pth'.format(args.model))
                    torch.save(encoder_model.state_dict(), save_mode_path)
                    torch.save(encoder_model.state_dict(), save_best_path)
                    logging.info("save best model 1 to {}".format(save_mode_path))
                encoder_model.train()

                decoder_model.eval()
                dice_sample_2 = test_3d_patch.var_all_case_LA(
                    args.root_path, decoder_model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if dice_sample_2 > best_dice_2:
                    best_dice_2 = round(dice_sample_2, 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_model2_dice_{}.pth'.format(iter_num, best_dice_2))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(decoder_model.state_dict(), save_mode_path)
                    torch.save(decoder_model.state_dict(), save_best_path)
                    logging.info("save best model 2 to {}".format(save_mode_path))
                decoder_model.train()

                inference_model.decoder.load_state_dict(encoder_model.decoder.state_dict())
                inference_model.encoder.load_state_dict(decoder_model.encoder.state_dict())
                inference_model.eval()

                dice_sample = test_3d_patch.var_all_case_LA(
                    args.root_path, inference_model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_model_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(inference_model.state_dict(), save_mode_path)
                    torch.save(inference_model.state_dict(), save_best_path)
                    logging.info("save best inference model to {}".format(save_mode_path))

                logging.info("iteration %d, dice 1: %.04f, dice 2: %.04f, inference dice: %.04f"
                             % (iter_num, dice_sample_1, dice_sample_2, dice_sample))

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break


# ============================================================
# Main (unchanged except file name copy)
# ============================================================

if __name__ == "__main__":
    pre_snapshot_path = "/content/drive/MyDrive/Research/models/DualMatch/{}/ours/{}_{}_labeled/pre_train".format(
        dataset, args.exp, args.labelnum
    )
    self_snapshot_path = "/content/drive/MyDrive/Research/models/DualMatch/{}/ours/{}_{}_labeled/self_train".format(
        dataset, args.exp, args.labelnum
    )

    print("Starting training with Uncertainty-Aware Duet + UA CutMix/CPS refinement.")

    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')

    # Copy THIS script into snapshot (update path to match your environment)
    # If you run in Colab, you can set the correct path here.
    try:
        shutil.copy('/content/drive/MyDrive/Research/DuetMatch/update_duatmatch.py', self_snapshot_path)
    except Exception as e:
        print(f"[Warning] Could not copy source file: {e}")

    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    pre_train(args, pre_snapshot_path)
    self_train(args, pre_snapshot_path, self_snapshot_path)
