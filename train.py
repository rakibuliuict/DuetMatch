# from asyncore import write
# # import imp

import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np
from skimage.measure import label

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.data import DataLoader

from utils import losses, ramps, test_3d_patch
from dataloaders.brats19 import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/Research/Dataset/brats2019', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='test', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=1000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=6000, help='maximum self-train iteration to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=10, help='trained samples')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default=10.0, help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')

# NEW: confidence threshold for pseudo-label filtering
parser.add_argument('--pl_conf_thres', type=float, default=0.7, help='confidence threshold for pseudo-label filtering (0.6~0.8)')

args = parser.parse_args()

dataset = 'test'
if 'brats2019' in args.root_path:
    dataset = 'brats2019'
elif 'brats2018' in args.root_path:
    dataset = 'brats2018'
elif 'brats2017' in args.root_path:
    dataset = 'brats2017'

def LargestCC_3d(segmentation):
    """
    segmentation: (B, D, H, W) tensor, binary 0/1
    Returns largest connected component per sample (still binary).
    Note: skimage.measure.label is n-dim, so it works for 3D.
    """
    N = segmentation.shape[0]
    out_list = []
    for n in range(N):
        arr = segmentation[n].detach().cpu().numpy().astype(np.int32)
        lab = label(arr)
        if lab.max() != 0:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            largest_id = counts.argmax()
            largest = (lab == largest_id).astype(np.float32)
        else:
            largest = arr.astype(np.float32)
        out_list.append(largest)
    return torch.from_numpy(np.stack(out_list, axis=0)).to(segmentation.device)

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

def labeled_ratio_to_patients(dataset, patiens_num):
    ref_dict = None
    if "brats2019" in dataset:
        ref_dict = {"4": 10, "10": 25, "20": 50}
    elif "brats2018" in dataset:
        ref_dict = {"10": 20}
    elif "brats2017" in dataset:
        ref_dict = {"10": 20}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

# ---------- NEW: 3D-correct pseudo-label + confidence ----------
def get_plab_and_conf(logits):
    """
    logits: (B, C, D, H, W)
    returns:
      plab: (B, D, H, W) long (argmax)
      conf: (B, D, H, W) float in [0,1] (max prob)
      prob_fg: (B, D, H, W) float = P(class=1) if binary else None-like behavior
    """
    probs = F.softmax(logits, dim=1)
    conf, plab = torch.max(probs, dim=1)      # both (B, D, H, W)
    prob_fg = probs[:, 1, ...] if probs.size(1) > 1 else conf
    return plab.long(), conf, prob_fg

# ---------- NEW: voxelwise KL distillation map ----------
def voxelwise_kl(student_logits, teacher_prob):
    """
    student_logits: (B,C,D,H,W)
    teacher_prob:   (B,C,D,H,W) soft targets
    returns:        (B,D,H,W) KL per voxel
    """
    log_s = F.log_softmax(student_logits, dim=1)
    kl = F.kl_div(log_s, teacher_prob, reduction='none')  # (B,C,D,H,W)
    kl = kl.sum(dim=1)                                   # (B,D,H,W)
    return kl

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

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
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'][:args.labeled_bs].cuda()
            label_batch  = sampled_batch['label'][:args.labeled_bs].cuda()

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2.0

            iter_num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' %
                         (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(
                    args.root_path, model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=64,
                    stride_z=64
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

def mix_consecutive_pairs(tensor, mask):
    """
    tensor: (B, ...)
    mask:   broadcastable to tensor shape; typically (B,1,D,H,W) or (B,D,H,W)
    Mix pairs (0,1), (2,3), ...
    """
    B = tensor.size(0)
    assert B % 2 == 0, "Batch size must be even for consecutive pair mixing."
    t = tensor.view(B // 2, 2, *tensor.shape[1:])
    m = mask.view(B // 2, 2, *mask.shape[1:]) if mask.size(0) == B else mask
    mixed = t * m + t.flip(1) * (1 - m)
    return mixed.view(B, *tensor.shape[2:])

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
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    DICE = losses.mask_DiceLoss(nclass=num_classes)
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

    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch  = sampled_batch['label'].cuda()

            img   = volume_batch[:args.labeled_bs]
            lab   = label_batch[:args.labeled_bs]
            unimg = volume_batch[args.labeled_bs:]

            # -------- labeled supervised losses ----------
            lb_output_encoder_model, _ = encoder_model(img)
            lb_output_decoder_model, _ = decoder_model(img)

            lb_loss_encoder_model = 0.5 * F.cross_entropy(lb_output_encoder_model, lab) + 0.5 * DICE(lb_output_encoder_model, lab)
            lb_loss_decoder_model = 0.5 * F.cross_entropy(lb_output_decoder_model, lab) + 0.5 * DICE(lb_output_decoder_model, lab)

            # -------- unlabeled duet-style losses ----------
            # feature consistency: encoder feature (input dropout) match decoder feature (clean)
            with torch.no_grad():
                _, ulb_feature_decoder_clean = decoder_model(unimg)

            _, ulb_feature_encoder_dropin = encoder_model(unimg, input_perturb='dropout', p=0.6)
            ulb_loss_encoder_model = F.mse_loss(ulb_feature_encoder_dropin, ulb_feature_decoder_clean)

            # prediction consistency for decoder using KL distillation + confidence mask from encoder (teacher)
            with torch.no_grad():
                t_logits, _ = encoder_model(unimg)                       # teacher logits (clean)
                t_prob = F.softmax(t_logits, dim=1)                     # (B,C,D,H,W)
                _, t_conf, _ = get_plab_and_conf(t_logits)              # (B,D,H,W)
                conf_mask_ulb = (t_conf > args.pl_conf_thres).float()   # (B,D,H,W)

            s_logits, ulb_feature_decoder_drop = decoder_model(unimg, feature_perturb='dropout', p=0.6)
            kl_map = voxelwise_kl(s_logits, t_prob)                     # (B,D,H,W)
            ulb_loss_decoder_model = (kl_map * conf_mask_ulb).sum() / (conf_mask_ulb.sum() + 1e-6)

            loss_encoder = lb_loss_encoder_model + 0.5 * ulb_loss_encoder_model
            loss_decoder = lb_loss_decoder_model + ulb_loss_decoder_model

            # -------- build inference model + pseudo labels + masks ----------
            with torch.no_grad():
                inference_model.decoder.load_state_dict(encoder_model.decoder.state_dict())
                inference_model.encoder.load_state_dict(decoder_model.encoder.state_dict())

                ulb_output_encoder_model, _ = encoder_model(unimg)
                ulb_output_decoder_model, _ = decoder_model(unimg)
                ulb_output_inference_model, _ = inference_model(unimg)

                plab_ulb_encoder_model, conf_enc, _ = get_plab_and_conf(ulb_output_encoder_model)
                plab_ulb_decoder_model, conf_dec, _ = get_plab_and_conf(ulb_output_decoder_model)
                plab_ulb_inference_model, conf_inf, _ = get_plab_and_conf(ulb_output_inference_model)

                # Context mask (returns masks compatible with unimg)
                img_mask, loss_mask = context_mask(unimg, args.mask_ratio)

                # confidence gate from inference model (stable path)
                conf_gate = (conf_inf > args.pl_conf_thres).float()     # (B,D,H,W)

            # -------- pairwise mixing (your "pairwise CutMix-like" consecutive mixing) ----------
            mixed_img = mix_consecutive_pairs(unimg, img_mask)

            # mix pseudo-labels the same way
            plab_mixed_encoder_model = mix_consecutive_pairs(plab_ulb_encoder_model, img_mask)
            plab_mixed_decoder_model = mix_consecutive_pairs(plab_ulb_decoder_model, img_mask)

            # agreement masks + confidence gating (more reliable than agreement alone)
            agree_enc = (plab_ulb_encoder_model == plab_ulb_inference_model).float()
            agree_dec = (plab_ulb_decoder_model == plab_ulb_inference_model).float()

            filtered_mask_encoder = agree_enc * conf_gate
            filtered_mask_decoder = agree_dec * conf_gate

            # mix the reliability masks consistently with loss_mask (your original logic)
            # loss_mask is typically (B,1,D,H,W) or (B,D,H,W); broadcast is okay.
            mixed_mask_encoder = filtered_mask_encoder * loss_mask + filtered_mask_encoder.flip(0) * (1 - loss_mask)
            mixed_mask_decoder = filtered_mask_decoder * loss_mask + filtered_mask_decoder.flip(0) * (1 - loss_mask)

            # -------- CPS / Cross guidance on mixed image ----------
            mixed_output_encoder_model, _ = encoder_model(mixed_img)
            mixed_output_decoder_model, _ = decoder_model(mixed_img)

            # CE expects target (B,D,H,W)
            loss_cps_encoder_map = CE(mixed_output_encoder_model, plab_mixed_decoder_model.long())  # (B,D,H,W)
            loss_cps_decoder_map = CE(mixed_output_decoder_model, plab_mixed_encoder_model.long())

            loss_cps_encoder = (loss_cps_encoder_map * mixed_mask_decoder).sum() / (mixed_mask_decoder.sum() + 1e-16)
            loss_cps_decoder = (loss_cps_decoder_map * mixed_mask_encoder).sum() / (mixed_mask_encoder.sum() + 1e-16)
            loss_cps = loss_cps_encoder + loss_cps_decoder

            # -------- optimize ----------
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = loss_encoder + loss_decoder + 0.5 * loss_cps
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            iter_num += 1
            logging.info('iteration %d : loss encoder: %03f, loss decoder: %03f, loss cps: %03f' %
                         (iter_num, loss_encoder.item(), loss_decoder.item(), loss_cps.item()))

            # EMA cross-update (your original idea)
            update_ema_variables(encoder_model.encoder, decoder_model.encoder, 0.99)
            update_ema_variables(decoder_model.decoder, encoder_model.decoder, 0.99)

            # -------- eval/checkpoint ----------
            if iter_num % 200 == 0:
                encoder_model.eval()
                dice_sample_1 = test_3d_patch.var_all_case_LA(
                    args.root_path, encoder_model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=64,
                    stride_z=64
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
                    args.root_path, decoder_model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=64,
                    stride_z=64
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
                    args.root_path, inference_model,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=64,
                    stride_z=64
                )
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_model_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(inference_model.state_dict(), save_mode_path)
                    torch.save(inference_model.state_dict(), save_best_path)
                    logging.info("save best inference model to {}".format(save_mode_path))

                logging.info("iteration %d, dice 1: %.04f, dice 2: %.04f, inference dice: %.04f" %
                             (iter_num, dice_sample_1, dice_sample_2, dice_sample))

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break

if __name__ == "__main__":
    pre_snapshot_path = "/content/drive/MyDrive/Research/models/DualMatch/{}/ours/{}_{}_labeled/pre_train".format(
        dataset, args.exp, args.labelnum
    )
    self_snapshot_path = "/content/drive/MyDrive/Research/models/DualMatch/{}/ours/{}_{}_labeled/self_train".format(
        dataset, args.exp, args.labelnum
    )

    print("Starting training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')

    shutil.copy('/content/drive/MyDrive/Research/DuetMatch/train.py', self_snapshot_path)

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
