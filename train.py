# import os
# import sys
# from tqdm import tqdm
# import shutil
# import argparse
# import logging
# import random
# import numpy as np

# import torch
# import torch.optim as optim
# from torchvision import transforms
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from utils import losses, ramps, test_3d_patch
# from dataloaders.brats19 import *
# from networks.net_factory import net_factory
# from utils.BCP_utils import context_mask


# # ------------------------- Args -------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/Research/Dataset/brats2019', help='Dataset root')
# parser.add_argument('--exp', type=str, default='test', help='Experiment name')
# parser.add_argument('--model', type=str, default='VNet', help='Model name')
# parser.add_argument('--pre_max_iteration', type=int, default=1000, help='Max pre-train iterations')
# parser.add_argument('--self_max_iteration', type=int, default=6000, help='Max self-train iterations')
# parser.add_argument('--labeled_bs', type=int, default=4, help='Labeled batch size')
# parser.add_argument('--batch_size', type=int, default=8, help='Total batch size')
# parser.add_argument('--base_lr', type=float, default=0.01, help='Learning rate')
# parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
# parser.add_argument('--labelnum', type=int, default=10, help='Labeled ratio index (4/10/20 like your mapping)')
# parser.add_argument('--gpu', type=str, default='1', help='GPU id')
# parser.add_argument('--seed', type=int, default=1337, help='Seed')

# # Run control
# parser.add_argument('--skip_pretrain', action='store_true', help='Skip pretraining and start self training from saved best pretrain model')
# parser.add_argument('--pretrained_ckpt', type=str, default='', help='Optional: explicit checkpoint path to start self-train')

# # Pretrain improvements
# parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal gamma')
# parser.add_argument('--focal_alpha', type=float, default=0.75, help='Focal alpha for positive class')
# parser.add_argument('--lambda_boundary', type=float, default=0.1, help='Boundary loss weight in pretraining')
# parser.add_argument('--lambda_dice', type=float, default=0.5, help='Dice weight in pretraining')
# parser.add_argument('--lambda_focal', type=float, default=0.5, help='Focal weight in pretraining')

# # SSL improvements (self-train)
# parser.add_argument('--mask_ratio', type=float, default=2/3, help='Ratio of context mask')

# parser.add_argument('--pl_conf_thres', type=float, default=0.7, help='Confidence threshold (0.6~0.8)')
# parser.add_argument('--temp', type=float, default=2.0, help='Distillation temperature (>=1.0)')
# parser.add_argument('--entropy_gamma', type=float, default=2.0, help='Exponent for entropy-based weights (>=1)')
# parser.add_argument('--lambda_kl', type=float, default=1.0, help='Weight for KL consistency term')
# parser.add_argument('--lambda_dice_u', type=float, default=0.5, help='Weight for soft Dice consistency on unlabeled')
# parser.add_argument('--lambda_cps', type=float, default=0.5, help='Weight for CPS term')
# parser.add_argument('--u_rampup_iters', type=int, default=1500, help='Rampup iterations for unlabeled losses')

# args = parser.parse_args()


# # ------------------------- Dataset name -------------------------
# dataset = 'test'
# if 'brats2019' in args.root_path:
#     dataset = 'brats2019'
# elif 'brats2018' in args.root_path:
#     dataset = 'brats2018'
# elif 'brats2017' in args.root_path:
#     dataset = 'brats2017'


# # ------------------------- Reproducibility -------------------------
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# if args.deterministic:
#     cudnn.benchmark = False
#     cudnn.deterministic = True
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     random.seed(args.seed)
#     np.random.seed(args.seed)


# # ------------------------- Globals -------------------------
# train_data_path = args.root_path
# patch_size = (96, 96, 96)
# num_classes = 2


# # ------------------------- Utils -------------------------
# def labeled_ratio_to_patients(dataset_path, patiens_num):
#     ref_dict = None
#     if "brats2019" in dataset_path:
#         ref_dict = {"4": 10, "10": 25, "20": 50}
#     elif "brats2018" in dataset_path:
#         ref_dict = {"10": 20}
#     elif "brats2017" in dataset_path:
#         ref_dict = {"10": 20}
#     else:
#         raise ValueError("Unknown dataset mapping for labeled_ratio_to_patients")
#     return ref_dict[str(patiens_num)]


# def save_net_opt(net, optimizer, path):
#     state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
#     torch.save(state, str(path))


# def load_net(net, path):
#     state = torch.load(str(path), map_location='cpu')
#     if isinstance(state, dict) and 'net' in state:
#         net.load_state_dict(state['net'])
#     else:
#         net.load_state_dict(state)


# @torch.no_grad()
# def update_ema_variables(model, ema_model, alpha):
#     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#         ema_param.data.mul_(alpha).add_((1.0 - alpha) * param.data)


# def get_plab_conf_probs(logits):
#     probs = F.softmax(logits, dim=1)
#     conf, plab = torch.max(probs, dim=1)  # (B,D,H,W)
#     return plab.long(), conf, probs


# def voxelwise_kl_temp(student_logits, teacher_logits, T=2.0):
#     t_prob = F.softmax(teacher_logits / T, dim=1)     # (B,C,D,H,W)
#     log_s = F.log_softmax(student_logits / T, dim=1)  # (B,C,D,H,W)
#     kl = F.kl_div(log_s, t_prob, reduction='none')    # (B,C,D,H,W)
#     kl = kl.sum(dim=1) * (T * T)                      # (B,D,H,W)
#     return kl


# def entropy_weight(probs, gamma=2.0):
#     eps = 1e-8
#     C = probs.size(1)
#     ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)  # (B,D,H,W)
#     ent_norm = ent / np.log(C)
#     w = (1.0 - ent_norm).clamp(0.0, 1.0) ** gamma
#     return w


# def soft_dice_consistency(student_probs, teacher_probs, weight_map=None, eps=1e-6):
#     B, C, D, H, W = student_probs.shape
#     s = student_probs.view(B, C, -1)
#     t = teacher_probs.view(B, C, -1)
#     if weight_map is not None:
#         w = weight_map.view(B, 1, -1)
#         s = s * w
#         t = t * w
#     inter = (s * t).sum(dim=2)
#     denom = s.sum(dim=2) + t.sum(dim=2)
#     dice = (2.0 * inter + eps) / (denom + eps)
#     return 1.0 - dice.mean()


# def rampup_factor(iter_num, rampup_iters):
#     if rampup_iters <= 0:
#         return 1.0
#     x = np.clip(iter_num / float(rampup_iters), 0.0, 1.0)
#     return float(ramps.sigmoid_rampup(x * rampup_iters, rampup_iters))


# def _prepare_pair_mask(mask, B, spatial, device, dtype=torch.float32):
#     """
#     Returns mask shaped (B, 1, D, H, W) float in [0,1], broadcastable to images/logits.
#     Accepts mask in (D,H,W), (1,D,H,W), (B,D,H,W), (B,1,D,H,W).
#     """
#     m = mask
#     if not torch.is_tensor(m):
#         m = torch.tensor(m, device=device, dtype=dtype)
#     else:
#         m = m.to(device=device, dtype=dtype)

#     if m.dim() == 3 and tuple(m.shape) == tuple(spatial):
#         m = m.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
#     elif m.dim() == 4 and tuple(m.shape[-3:]) == tuple(spatial) and m.size(0) == 1:
#         m = m.unsqueeze(1)               # (1,1,D,H,W)
#     elif m.dim() == 4 and tuple(m.shape[-3:]) == tuple(spatial) and m.size(0) == B:
#         m = m.unsqueeze(1)               # (B,1,D,H,W)
#     elif m.dim() == 5 and tuple(m.shape[-3:]) == tuple(spatial):
#         pass
#     else:
#         raise RuntimeError(f"Unexpected mask shape {tuple(m.shape)} for spatial {spatial} and batch {B}")

#     if m.size(0) == 1 and B > 1:
#         m = m.expand(B, *m.shape[1:])    # (B,1,D,H,W)

#     return m


# def mix_images_pairs(x, mask):
#     """
#     x: (B, C, D, H, W) float
#     mask: broadcastable; uses soft mixing x0*m + x1*(1-m)
#     """
#     B = x.size(0)
#     assert B % 2 == 0, "Batch size must be even."
#     spatial = x.shape[-3:]
#     m = _prepare_pair_mask(mask, B, spatial, x.device, dtype=x.dtype)  # (B,1,D,H,W)

#     out = x.clone()
#     idx0 = torch.arange(0, B, 2, device=x.device)
#     idx1 = idx0 + 1

#     out[idx0] = x[idx0] * m[idx0] + x[idx1] * (1.0 - m[idx0])
#     out[idx1] = x[idx1] * m[idx1] + x[idx0] * (1.0 - m[idx1])
#     return out


# def mix_labels_pairs(y, mask):
#     """
#     y: (B, D, H, W) long  (or (B,1,D,H,W) long -> will squeeze)
#     mask: CutMix selection. No interpolation. Output stays long.
#     """
#     if y.dim() == 5 and y.size(1) == 1:
#         y = y.squeeze(1)

#     B = y.size(0)
#     assert B % 2 == 0, "Batch size must be even."
#     spatial = y.shape[-3:]
#     m = _prepare_pair_mask(mask, B, spatial, y.device, dtype=torch.float32)  # float mask
#     m = (m[:, 0] >= 0.5)  # (B,D,H,W) boolean selection

#     out = y.clone()
#     idx0 = torch.arange(0, B, 2, device=y.device)
#     idx1 = idx0 + 1

#     out[idx0] = torch.where(m[idx0], y[idx0], y[idx1])
#     out[idx1] = torch.where(m[idx1], y[idx1], y[idx0])
#     return out


# # ------------------------- Pretrain Improvements -------------------------
# def focal_loss_multiclass(logits, target, alpha_pos=0.75, gamma=2.0, eps=1e-8):
#     logp = F.log_softmax(logits, dim=1)
#     p = torch.exp(logp)

#     t = target.unsqueeze(1)
#     p_t = torch.gather(p, dim=1, index=t).squeeze(1).clamp_min(eps)
#     logp_t = torch.gather(logp, dim=1, index=t).squeeze(1)

#     if logits.size(1) == 2:
#         alpha = torch.where(
#             target == 1,
#             torch.tensor(alpha_pos, device=logits.device),
#             torch.tensor(1.0 - alpha_pos, device=logits.device)
#         )
#     else:
#         alpha = torch.ones_like(p_t)

#     loss = -alpha * (1.0 - p_t) ** gamma * logp_t
#     return loss.mean()


# def morphological_boundary_map(mask_float, k=3):
#     pad = k // 2
#     dil = F.max_pool3d(mask_float, kernel_size=k, stride=1, padding=pad)
#     ero = -F.max_pool3d(-mask_float, kernel_size=k, stride=1, padding=pad)
#     grad = (dil - ero).clamp(0.0, 1.0)
#     return grad


# def boundary_loss_from_logits(logits, target):
#     probs = F.softmax(logits, dim=1)[:, 1:2, ...]
#     gt = (target == 1).float().unsqueeze(1)
#     b_pred = morphological_boundary_map(probs, k=3)
#     b_gt = morphological_boundary_map(gt, k=3)
#     return F.l1_loss(b_pred, b_gt)


# # ------------------------- Pre-train -------------------------
# def pre_train(args, snapshot_path):
#     model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

#     db_train = BraTS(
#         base_dir=train_data_path,
#         split='train',
#         transform=transforms.Compose([
#             RandomRotFlip(),
#             RandomCrop(patch_size),
#             ToTensor(),
#         ])
#     )
#     logging.info(f'Max samples: {len(db_train)}')

#     labelnum = labeled_ratio_to_patients(args.root_path, args.labelnum)
#     labeled_idxs = list(range(labelnum))
#     unlabeled_idxs = list(range(labelnum, len(db_train)))

#     batch_sampler = TwoStreamBatchSampler(
#         labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
#     )

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(
#         db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn
#     )

#     optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
#     DICE = losses.mask_DiceLoss(nclass=num_classes)

#     model.train()
#     iter_num = 0
#     best_dice = 0
#     max_epoch = args.pre_max_iteration // len(trainloader) + 1
#     iterator = tqdm(range(max_epoch), ncols=70)

#     for _ in iterator:
#         for _, batch in enumerate(trainloader):
#             volume = batch['image'][:args.labeled_bs].cuda()
#             target = batch['label'][:args.labeled_bs].cuda()

#             logits, _ = model(volume)

#             loss_focal = focal_loss_multiclass(logits, target, alpha_pos=args.focal_alpha, gamma=args.focal_gamma)
#             loss_dice = DICE(logits, target)
#             loss_bd = boundary_loss_from_logits(logits, target)

#             loss = args.lambda_focal * loss_focal + args.lambda_dice * loss_dice + args.lambda_boundary * loss_bd

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             iter_num += 1
#             logging.info(
#                 f'pre iter {iter_num} | loss={loss.item():.4f} | focal={loss_focal.item():.4f} | dice={loss_dice.item():.4f} | bd={loss_bd.item():.4f}'
#             )

#             if iter_num % 200 == 0:
#                 model.eval()
#                 dice_sample = test_3d_patch.var_all_case_LA(
#                     args.root_path, model, num_classes=num_classes, patch_size=patch_size, stride_xy=64, stride_z=64
#                 )
#                 if dice_sample > best_dice:
#                     best_dice = round(dice_sample, 4)
#                     save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
#                     save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
#                     save_net_opt(model, optimizer, save_mode_path)
#                     save_net_opt(model, optimizer, save_best_path)
#                     logging.info(f"Saved best pretrain model -> {save_best_path}")
#                 model.train()
#                 logging.info(f"pre iter {iter_num} | val dice={dice_sample:.4f}")

#             if iter_num >= args.pre_max_iteration:
#                 iterator.close()
#                 return


# # ------------------------- Self-train -------------------------
# def self_train(args, pre_snapshot_path, self_snapshot_path):
#     encoder_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
#     decoder_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
#     inference_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

#     if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
#         pretrained_path = args.pretrained_ckpt
#     else:
#         pretrained_path = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')

#     load_net(encoder_model, pretrained_path)
#     load_net(decoder_model, pretrained_path)

#     for p in encoder_model.decoder.parameters():
#         p.requires_grad = False
#     for p in decoder_model.encoder.parameters():
#         p.requires_grad = False

#     db_train = BraTS(
#         base_dir=train_data_path,
#         split='train',
#         transform=transforms.Compose([
#             RandomRotFlip(),
#             RandomCrop(patch_size),
#             ToTensor(),
#         ])
#     )

#     labelnum = labeled_ratio_to_patients(args.root_path, args.labelnum)
#     labeled_idxs = list(range(labelnum))
#     unlabeled_idxs = list(range(labelnum, len(db_train)))

#     batch_sampler = TwoStreamBatchSampler(
#         labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
#     )

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(
#         db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn
#     )

#     DICE = losses.mask_DiceLoss(nclass=num_classes)
#     CE = nn.CrossEntropyLoss(reduction='none')

#     enc_opt = optim.SGD(encoder_model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
#     dec_opt = optim.SGD(decoder_model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

#     encoder_model.train()
#     decoder_model.train()

#     iter_num = 0
#     best_dice_inf = 0
#     best_dice_1 = 0
#     best_dice_2 = 0

#     max_epoch = args.self_max_iteration // len(trainloader) + 1
#     iterator = tqdm(range(max_epoch), ncols=70)

#     for _ in iterator:
#         for _, batch in enumerate(trainloader):
#             volume = batch['image'].cuda()
#             target = batch['label'].cuda()

#             img = volume[:args.labeled_bs]
#             lab = target[:args.labeled_bs]
#             unimg = volume[args.labeled_bs:]

#             lb_logits_enc, _ = encoder_model(img)
#             lb_logits_dec, _ = decoder_model(img)

#             lb_loss_enc = 0.5 * F.cross_entropy(lb_logits_enc, lab) + 0.5 * DICE(lb_logits_enc, lab)
#             lb_loss_dec = 0.5 * F.cross_entropy(lb_logits_dec, lab) + 0.5 * DICE(lb_logits_dec, lab)

#             with torch.no_grad():
#                 t_logits, _ = encoder_model(unimg)
#                 _, t_conf, t_probs = get_plab_conf_probs(t_logits)
#                 w_conf = (t_conf > args.pl_conf_thres).float()
#                 w_ent = entropy_weight(t_probs, gamma=args.entropy_gamma)
#                 w = w_conf * w_ent

#             s_logits_dec, _ = decoder_model(unimg, feature_perturb='dropout', p=0.6)

#             kl_map = voxelwise_kl_temp(s_logits_dec, t_logits, T=args.temp)
#             u_kl = (kl_map * w).sum() / (w.sum() + 1e-6)

#             s_probs = F.softmax(s_logits_dec, dim=1)
#             t_probs_T1 = F.softmax(t_logits, dim=1)
#             u_dice = soft_dice_consistency(s_probs, t_probs_T1, weight_map=w)

#             with torch.no_grad():
#                 _, feat_dec_clean = decoder_model(unimg)
#             _, feat_enc_dropin = encoder_model(unimg, input_perturb='dropout', p=0.6)
#             u_feat = F.mse_loss(feat_enc_dropin, feat_dec_clean)

#             ru = rampup_factor(iter_num, args.u_rampup_iters)
#             u_loss_total = ru * (args.lambda_kl * u_kl + args.lambda_dice_u * u_dice)

#             loss_enc = lb_loss_enc + 0.5 * u_feat
#             loss_dec = lb_loss_dec + u_loss_total

#             with torch.no_grad():
#                 inference_model.decoder.load_state_dict(encoder_model.decoder.state_dict())
#                 inference_model.encoder.load_state_dict(decoder_model.encoder.state_dict())
#                 inf_logits, _ = inference_model(unimg)
#                 inf_plab, inf_conf, inf_probs = get_plab_conf_probs(inf_logits)

#                 img_mask, loss_mask = context_mask(unimg, args.mask_ratio)

#                 if torch.is_tensor(loss_mask):
#                     loss_mask = loss_mask.to(device=unimg.device, dtype=unimg.dtype)
#                 else:
#                     loss_mask = torch.tensor(loss_mask, device=unimg.device, dtype=unimg.dtype)

#                 if loss_mask.dim() == 5 and loss_mask.size(1) == 1:
#                     loss_mask = loss_mask.squeeze(1)
#                 elif loss_mask.dim() == 4 and loss_mask.size(0) == 1:
#                     loss_mask = loss_mask.expand(unimg.size(0), *loss_mask.shape[1:])
#                 elif loss_mask.dim() == 3:
#                     loss_mask = loss_mask.unsqueeze(0).expand(unimg.size(0), *loss_mask.shape)

#                 enc_logits_ulb, _ = encoder_model(unimg)
#                 dec_logits_ulb, _ = decoder_model(unimg)
#                 enc_plab_ulb, _, _ = get_plab_conf_probs(enc_logits_ulb)
#                 dec_plab_ulb, _, _ = get_plab_conf_probs(dec_logits_ulb)

#                 agree_enc = (enc_plab_ulb == inf_plab).float()
#                 agree_dec = (dec_plab_ulb == inf_plab).float()

#                 gate = (inf_conf > args.pl_conf_thres).float()
#                 gate = gate * entropy_weight(inf_probs, gamma=args.entropy_gamma)

#                 rel_mask_enc = agree_enc * gate
#                 rel_mask_dec = agree_dec * gate

#             mixed_img = mix_images_pairs(unimg, img_mask)
#             mixed_plab_enc = mix_labels_pairs(enc_plab_ulb, img_mask)
#             mixed_plab_dec = mix_labels_pairs(dec_plab_ulb, img_mask)

#             mixed_rel_enc = rel_mask_enc * loss_mask + rel_mask_enc.flip(0) * (1.0 - loss_mask)
#             mixed_rel_dec = rel_mask_dec * loss_mask + rel_mask_dec.flip(0) * (1.0 - loss_mask)

#             mixed_logits_enc, _ = encoder_model(mixed_img)
#             mixed_logits_dec, _ = decoder_model(mixed_img)

#             cps_map_enc = CE(mixed_logits_enc, mixed_plab_dec.long())
#             cps_map_dec = CE(mixed_logits_dec, mixed_plab_enc.long())

#             cps_enc = (cps_map_enc * mixed_rel_dec).sum() / (mixed_rel_dec.sum() + 1e-6)
#             cps_dec = (cps_map_dec * mixed_rel_enc).sum() / (mixed_rel_enc.sum() + 1e-6)
#             cps_loss = cps_enc + cps_dec
#             cps_loss = ru * args.lambda_cps * cps_loss

#             total_loss = loss_enc + loss_dec + cps_loss

#             enc_opt.zero_grad()
#             dec_opt.zero_grad()
#             total_loss.backward()
#             enc_opt.step()
#             dec_opt.step()

#             iter_num += 1

#             update_ema_variables(encoder_model.encoder, decoder_model.encoder, 0.99)
#             update_ema_variables(decoder_model.decoder, encoder_model.decoder, 0.99)

#             logging.info(
#                 f"self iter {iter_num} | "
#                 f"lb_enc={lb_loss_enc.item():.4f} lb_dec={lb_loss_dec.item():.4f} | "
#                 f"u_kl={u_kl.item():.4f} u_dice={u_dice.item():.4f} u_feat={u_feat.item():.4f} | "
#                 f"cps={cps_loss.item():.4f} | ru={ru:.3f} | total={total_loss.item():.4f}"
#             )

#             if iter_num % 200 == 0:
#                 encoder_model.eval()
#                 d1 = test_3d_patch.var_all_case_LA(
#                     args.root_path, encoder_model, num_classes=num_classes,
#                     patch_size=patch_size, stride_xy=64, stride_z=64
#                 )
#                 if d1 > best_dice_1:
#                     best_dice_1 = round(d1, 4)
#                     torch.save(encoder_model.state_dict(), os.path.join(self_snapshot_path, f'iter_{iter_num}_model1_dice_{best_dice_1}.pth'))
#                     torch.save(encoder_model.state_dict(), os.path.join(self_snapshot_path, f'{args.model}_best_model1.pth'))
#                     logging.info(f"Saved best model1 -> dice={best_dice_1}")
#                 encoder_model.train()

#                 decoder_model.eval()
#                 d2 = test_3d_patch.var_all_case_LA(
#                     args.root_path, decoder_model, num_classes=num_classes,
#                     patch_size=patch_size, stride_xy=64, stride_z=64
#                 )
#                 if d2 > best_dice_2:
#                     best_dice_2 = round(d2, 4)
#                     torch.save(decoder_model.state_dict(), os.path.join(self_snapshot_path, f'iter_{iter_num}_model2_dice_{best_dice_2}.pth'))
#                     torch.save(decoder_model.state_dict(), os.path.join(self_snapshot_path, f'{args.model}_best_model2.pth'))
#                     logging.info(f"Saved best model2 -> dice={best_dice_2}")
#                 decoder_model.train()

#                 inference_model.decoder.load_state_dict(encoder_model.decoder.state_dict())
#                 inference_model.encoder.load_state_dict(decoder_model.encoder.state_dict())
#                 inference_model.eval()
#                 di = test_3d_patch.var_all_case_LA(
#                     args.root_path, inference_model, num_classes=num_classes,
#                     patch_size=patch_size, stride_xy=64, stride_z=64
#                 )
#                 if di > best_dice_inf:
#                     best_dice_inf = round(di, 4)
#                     torch.save(inference_model.state_dict(), os.path.join(self_snapshot_path, f'iter_{iter_num}_inference_dice_{best_dice_inf}.pth'))
#                     torch.save(inference_model.state_dict(), os.path.join(self_snapshot_path, f'{args.model}_best_model.pth'))
#                     logging.info(f"Saved best inference -> dice={best_dice_inf}")

#                 logging.info(f"self iter {iter_num} | val dice: model1={d1:.4f} model2={d2:.4f} inference={di:.4f}")

#             if iter_num >= args.self_max_iteration:
#                 iterator.close()
#                 return


# # ------------------------- Main -------------------------
# if __name__ == "__main__":
#     pre_snapshot_path = f"/content/drive/MyDrive/Research/models/DualMatch/{dataset}/ours/{args.exp}_{args.labelnum}_labeled/pre_train"
#     self_snapshot_path = f"/content/drive/MyDrive/Research/models/DualMatch/{dataset}/ours/{args.exp}_{args.labelnum}_labeled/self_train"

#     print("Starting training.")
#     for sp in [pre_snapshot_path, self_snapshot_path]:
#         os.makedirs(sp, exist_ok=True)
#         if os.path.exists(sp + '/code'):
#             shutil.rmtree(sp + '/code')

#     try:
#         shutil.copy('/content/drive/MyDrive/Research/DuetMatch/train.py', self_snapshot_path)
#     except Exception:
#         pass

#     logging.basicConfig(
#         filename=pre_snapshot_path + "/log.txt",
#         level=logging.INFO,
#         format='[%(asctime)s.%(msecs)03d] %(message)s',
#         datefmt='%H:%M:%S'
#     )
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))

#     if not args.skip_pretrain:
#         pre_train(args, pre_snapshot_path)

#     self_train(args, pre_snapshot_path, self_snapshot_path)



import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np

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


# ------------------------- Args -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/Research/Dataset/brats2019', help='Dataset root')
parser.add_argument('--exp', type=str, default='test', help='Experiment name')
parser.add_argument('--model', type=str, default='VNet', help='Model name')
parser.add_argument('--pre_max_iteration', type=int, default=1000, help='Max pre-train iterations')
parser.add_argument('--self_max_iteration', type=int, default=6000, help='Max self-train iterations')
parser.add_argument('--labeled_bs', type=int, default=4, help='Labeled batch size')
parser.add_argument('--batch_size', type=int, default=8, help='Total batch size')
parser.add_argument('--base_lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
parser.add_argument('--labelnum', type=int, default=10, help='Labeled ratio index (4/10/20 like your mapping)')
parser.add_argument('--gpu', type=str, default='1', help='GPU id')
parser.add_argument('--seed', type=int, default=1337, help='Seed')

# Run control
parser.add_argument('--skip_pretrain', action='store_true', help='Skip pretraining and start self training from saved best pretrain model')
parser.add_argument('--pretrained_ckpt', type=str, default='', help='Optional: explicit checkpoint path to start self-train')

# Pretrain improvements
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal gamma')
parser.add_argument('--focal_alpha', type=float, default=0.75, help='Focal alpha for positive class')
parser.add_argument('--lambda_boundary', type=float, default=0.1, help='Boundary loss weight in pretraining')
parser.add_argument('--lambda_dice', type=float, default=0.5, help='Dice weight in pretraining')
parser.add_argument('--lambda_focal', type=float, default=0.5, help='Focal weight in pretraining')

# SSL improvements (self-train)
parser.add_argument('--mask_ratio', type=float, default=2/3, help='Ratio of context mask')

# You can keep these (some are now less important, but harmless)
parser.add_argument('--pl_conf_thres', type=float, default=0.7, help='Confidence threshold (0.6~0.8)')
parser.add_argument('--temp', type=float, default=2.0, help='Distillation temperature (>=1.0)')
parser.add_argument('--entropy_gamma', type=float, default=2.0, help='Exponent for entropy-based weights (>=1)')
parser.add_argument('--lambda_kl', type=float, default=1.0, help='(unused after U-Duet) Weight for KL consistency term')
parser.add_argument('--lambda_dice_u', type=float, default=0.5, help='(unused after U-Duet) Weight for soft Dice consistency on unlabeled')
parser.add_argument('--lambda_cps', type=float, default=0.5, help='Weight for CPS term')
parser.add_argument('--u_rampup_iters', type=int, default=1500, help='Rampup iterations for unlabeled losses')

# NEW: Uncertainty-Aware Duet hyperparams
parser.add_argument('--u_duet_alpha_feat', type=float, default=0.5, help='alpha for feature consistency in U-Duet')
parser.add_argument('--u_duet_mu_ent', type=float, default=0.1, help='mu for entropy regularizer in U-Duet')
parser.add_argument('--beta_max', type=float, default=1.0, help='beta max for uncertainty schedule')
parser.add_argument('--beta_min', type=float, default=0.1, help='beta min for uncertainty schedule')
parser.add_argument('--beta_decay', type=float, default=0.1, help='beta decay for uncertainty schedule')

# NEW: Uncertainty-aware pseudo-label refinement
parser.add_argument('--pl_kappa', type=float, default=1.0, help='confidence sharpness exp(-kappa*entropy)')
parser.add_argument('--pl_refine_thres', type=float, default=0.5, help='if conf < thres, replace PL with stable PL')

args = parser.parse_args()


# ------------------------- Dataset name -------------------------
dataset = 'test'
if 'brats2019' in args.root_path:
    dataset = 'brats2019'
elif 'brats2018' in args.root_path:
    dataset = 'brats2018'
elif 'brats2017' in args.root_path:
    dataset = 'brats2017'


# ------------------------- Reproducibility -------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


# ------------------------- Globals -------------------------
train_data_path = args.root_path
patch_size = (96, 96, 96)
num_classes = 2


# ------------------------- Utils -------------------------
def labeled_ratio_to_patients(dataset_path, patiens_num):
    ref_dict = None
    if "brats2019" in dataset_path:
        ref_dict = {"4": 10, "10": 25, "20": 50}
    elif "brats2018" in dataset_path:
        ref_dict = {"10": 20}
    elif "brats2017" in dataset_path:
        ref_dict = {"10": 20}
    else:
        raise ValueError("Unknown dataset mapping for labeled_ratio_to_patients")
    return ref_dict[str(patiens_num)]


def save_net_opt(net, optimizer, path):
    state = {'net': net.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, str(path))


def load_net(net, path):
    state = torch.load(str(path), map_location='cpu')
    if isinstance(state, dict) and 'net' in state:
        net.load_state_dict(state['net'])
    else:
        net.load_state_dict(state)


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1.0 - alpha) * param.data)


def get_plab_conf_probs(logits):
    probs = F.softmax(logits, dim=1)
    conf, plab = torch.max(probs, dim=1)  # (B,D,H,W)
    return plab.long(), conf, probs


def entropy_from_probs(probs, eps=1e-8):
    # probs: (B,C,D,H,W) -> (B,1,D,H,W)
    H = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1, keepdim=True)
    return H


def beta_schedule(iter_num, max_iter, beta_max=1.0, beta_min=0.1, decay=0.1):
    t = float(iter_num)
    T = float(max_iter) + 1e-8
    beta = beta_max * np.exp(-decay * (t / T))
    return float(max(beta_min, beta))


def uncertainty_weight(Hs, Ht, beta_t):
    # Hs, Ht: (B,1,D,H,W), beta_t: scalar tensor
    return 1.0 / (torch.exp(beta_t * Hs) + torch.exp(beta_t * Ht) + 1e-8)  # (B,1,D,H,W)


def weighted_feature_mse(Fs, Ft, w):
    # Fs,Ft: (B,F,D,H,W), w:(B,1,D,H,W)
    mse_map = (Fs - Ft).pow(2).mean(dim=1, keepdim=True)  # (B,1,D,H,W)
    return (mse_map * w).mean()


def soft_ce_map(student_logits, teacher_probs):
    # student_logits: (B,C,D,H,W), teacher_probs: (B,C,D,H,W) -> returns (B,D,H,W)
    logp = F.log_softmax(student_logits, dim=1)
    loss = -(teacher_probs * logp).sum(dim=1)  # (B,D,H,W)
    return loss


def confidence_from_entropy(probs, kappa=1.0):
    # probs: (B,C,D,H,W) -> conf: (B,D,H,W) in (0,1]
    H = entropy_from_probs(probs)            # (B,1,D,H,W)
    conf = torch.exp(-kappa * H).squeeze(1) # (B,D,H,W)
    return conf


def rampup_factor(iter_num, rampup_iters):
    if rampup_iters <= 0:
        return 1.0
    x = np.clip(iter_num / float(rampup_iters), 0.0, 1.0)
    return float(ramps.sigmoid_rampup(x * rampup_iters, rampup_iters))


def _prepare_pair_mask(mask, B, spatial, device, dtype=torch.float32):
    """
    Returns mask shaped (B, 1, D, H, W) float in [0,1], broadcastable to images/logits.
    Accepts mask in (D,H,W), (1,D,H,W), (B,D,H,W), (B,1,D,H,W).
    """
    m = mask
    if not torch.is_tensor(m):
        m = torch.tensor(m, device=device, dtype=dtype)
    else:
        m = m.to(device=device, dtype=dtype)

    if m.dim() == 3 and tuple(m.shape) == tuple(spatial):
        m = m.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    elif m.dim() == 4 and tuple(m.shape[-3:]) == tuple(spatial) and m.size(0) == 1:
        m = m.unsqueeze(1)               # (1,1,D,H,W)
    elif m.dim() == 4 and tuple(m.shape[-3:]) == tuple(spatial) and m.size(0) == B:
        m = m.unsqueeze(1)               # (B,1,D,H,W)
    elif m.dim() == 5 and tuple(m.shape[-3:]) == tuple(spatial):
        pass
    else:
        raise RuntimeError(f"Unexpected mask shape {tuple(m.shape)} for spatial {spatial} and batch {B}")

    if m.size(0) == 1 and B > 1:
        m = m.expand(B, *m.shape[1:])    # (B,1,D,H,W)

    return m


def mix_images_pairs(x, mask):
    """
    x: (B, C, D, H, W) float
    mask: broadcastable; uses soft mixing x0*m + x1*(1-m)
    """
    B = x.size(0)
    assert B % 2 == 0, "Batch size must be even."
    spatial = x.shape[-3:]
    m = _prepare_pair_mask(mask, B, spatial, x.device, dtype=x.dtype)  # (B,1,D,H,W)

    out = x.clone()
    idx0 = torch.arange(0, B, 2, device=x.device)
    idx1 = idx0 + 1

    out[idx0] = x[idx0] * m[idx0] + x[idx1] * (1.0 - m[idx0])
    out[idx1] = x[idx1] * m[idx1] + x[idx0] * (1.0 - m[idx1])
    return out


def mix_labels_pairs(y, mask):
    """
    y: (B, D, H, W) long  (or (B,1,D,H,W) long -> will squeeze)
    mask: CutMix selection. No interpolation. Output stays long.
    """
    if y.dim() == 5 and y.size(1) == 1:
        y = y.squeeze(1)

    B = y.size(0)
    assert B % 2 == 0, "Batch size must be even."
    spatial = y.shape[-3:]
    m = _prepare_pair_mask(mask, B, spatial, y.device, dtype=torch.float32)  # float mask
    m = (m[:, 0] >= 0.5)  # (B,D,H,W) boolean selection

    out = y.clone()
    idx0 = torch.arange(0, B, 2, device=y.device)
    idx1 = idx0 + 1

    out[idx0] = torch.where(m[idx0], y[idx0], y[idx1])
    out[idx1] = torch.where(m[idx1], y[idx1], y[idx0])
    return out


# ------------------------- Pretrain Improvements -------------------------
def focal_loss_multiclass(logits, target, alpha_pos=0.75, gamma=2.0, eps=1e-8):
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)

    t = target.unsqueeze(1)
    p_t = torch.gather(p, dim=1, index=t).squeeze(1).clamp_min(eps)
    logp_t = torch.gather(logp, dim=1, index=t).squeeze(1)

    if logits.size(1) == 2:
        alpha = torch.where(
            target == 1,
            torch.tensor(alpha_pos, device=logits.device),
            torch.tensor(1.0 - alpha_pos, device=logits.device)
        )
    else:
        alpha = torch.ones_like(p_t)

    loss = -alpha * (1.0 - p_t) ** gamma * logp_t
    return loss.mean()


def morphological_boundary_map(mask_float, k=3):
    pad = k // 2
    dil = F.max_pool3d(mask_float, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool3d(-mask_float, kernel_size=k, stride=1, padding=pad)
    grad = (dil - ero).clamp(0.0, 1.0)
    return grad


def boundary_loss_from_logits(logits, target):
    probs = F.softmax(logits, dim=1)[:, 1:2, ...]
    gt = (target == 1).float().unsqueeze(1)
    b_pred = morphological_boundary_map(probs, k=3)
    b_gt = morphological_boundary_map(gt, k=3)
    return F.l1_loss(b_pred, b_gt)


# ------------------------- Pre-train -------------------------
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
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn
    )

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=num_classes)

    model.train()
    iter_num = 0
    best_dice = 0
    max_epoch = args.pre_max_iteration // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, batch in enumerate(trainloader):
            volume = batch['image'][:args.labeled_bs].cuda()
            target = batch['label'][:args.labeled_bs].cuda()

            logits, _ = model(volume)

            loss_focal = focal_loss_multiclass(logits, target, alpha_pos=args.focal_alpha, gamma=args.focal_gamma)
            loss_dice = DICE(logits, target)
            loss_bd = boundary_loss_from_logits(logits, target)

            loss = args.lambda_focal * loss_focal + args.lambda_dice * loss_dice + args.lambda_boundary * loss_bd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            logging.info(
                f'pre iter {iter_num} | loss={loss.item():.4f} | focal={loss_focal.item():.4f} | dice={loss_dice.item():.4f} | bd={loss_bd.item():.4f}'
            )

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(
                    args.root_path, model, num_classes=num_classes, patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, f'iter_{iter_num}_dice_{best_dice}.pth')
                    save_best_path = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info(f"Saved best pretrain model -> {save_best_path}")
                model.train()
                logging.info(f"pre iter {iter_num} | val dice={dice_sample:.4f}")

            if iter_num >= args.pre_max_iteration:
                iterator.close()
                return


# ------------------------- NEW: voxel-wise focal map for CPS -------------------------
def focal_map_from_logits(logits, target, alpha_pos=0.75, gamma=2.0, eps=1e-8):
    """
    logits: (B,C,D,H,W)
    target: (B,D,H,W) long
    returns: (B,D,H,W) focal loss map (no reduction)
    """
    logp = F.log_softmax(logits, dim=1)     # (B,C,D,H,W)
    p = torch.exp(logp)

    t = target.unsqueeze(1)                # (B,1,D,H,W)
    p_t = torch.gather(p, 1, t).squeeze(1).clamp_min(eps)        # (B,D,H,W)
    logp_t = torch.gather(logp, 1, t).squeeze(1)                 # (B,D,H,W)

    # binary alpha
    if logits.size(1) == 2:
        alpha = torch.where(
            target == 1,
            torch.tensor(alpha_pos, device=logits.device),
            torch.tensor(1.0 - alpha_pos, device=logits.device)
        )
    else:
        alpha = torch.ones_like(p_t)

    loss = -alpha * (1.0 - p_t) ** gamma * logp_t               # (B,D,H,W)
    return loss


# ------------------------- Self-train (UPDATED) -------------------------
def self_train(args, pre_snapshot_path, self_snapshot_path):
    encoder_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    decoder_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    inference_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        pretrained_path = args.pretrained_ckpt
    else:
        pretrained_path = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')

    load_net(encoder_model, pretrained_path)
    load_net(decoder_model, pretrained_path)

    # DuetMatch decoupling
    for p in encoder_model.decoder.parameters():
        p.requires_grad = False
    for p in decoder_model.encoder.parameters():
        p.requires_grad = False

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
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn
    )

    DICE = losses.mask_DiceLoss(nclass=num_classes)

    enc_opt = optim.SGD(encoder_model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dec_opt = optim.SGD(decoder_model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    encoder_model.train()
    decoder_model.train()

    iter_num = 0
    best_dice_inf = 0
    best_dice_1 = 0
    best_dice_2 = 0

    max_epoch = args.self_max_iteration // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, batch in enumerate(trainloader):
            volume = batch['image'].cuda()
            target = batch['label'].cuda()

            img = volume[:args.labeled_bs]
            lab = target[:args.labeled_bs]
            unimg = volume[args.labeled_bs:]

            # -----------------------------
            # (1) Supervised losses (unchanged)
            # -----------------------------
            lb_logits_enc, _ = encoder_model(img)
            lb_logits_dec, _ = decoder_model(img)

            lb_loss_enc = 0.5 * F.cross_entropy(lb_logits_enc, lab) + 0.5 * DICE(lb_logits_enc, lab)
            lb_loss_dec = 0.5 * F.cross_entropy(lb_logits_dec, lab) + 0.5 * DICE(lb_logits_dec, lab)

            # -----------------------------
            # (2) Uncertainty-Aware Duet Loss (KEY CONTRIBUTION)
            # Teacher: encoder_model (stable branch)
            # Student: decoder_model with dropout perturbation
            # -----------------------------
            with torch.no_grad():
                t_logits, t_feat = encoder_model(unimg)               # logits + features
                t_probs = F.softmax(t_logits, dim=1)                  # (B,C,D,H,W)
                H_t = entropy_from_probs(t_probs)                     # (B,1,D,H,W)

            s_logits, s_feat = decoder_model(unimg, feature_perturb='dropout', p=0.6)
            s_probs = F.softmax(s_logits, dim=1)
            H_s = entropy_from_probs(s_probs)

            beta_val = beta_schedule(iter_num, args.self_max_iteration, args.beta_max, args.beta_min, args.beta_decay)
            beta_t = torch.tensor(beta_val, device=unimg.device, dtype=torch.float32)

            w_u = uncertainty_weight(H_s, H_t, beta_t)                # (B,1,D,H,W)

            # feature consistency (weighted)
            L_F = weighted_feature_mse(s_feat, t_feat.detach(), w_u)

            # prediction consistency (weighted soft CE)
            ce_map = soft_ce_map(s_logits, t_probs.detach())           # (B,D,H,W)
            L_P = (ce_map.unsqueeze(1) * w_u).mean()

            # entropy regularizer
            L_H = (H_s.mean() + H_t.mean())

            L_u_duet = args.u_duet_alpha_feat * L_F + L_P + args.u_duet_mu_ent * beta_t * L_H

            # ramp-up on unlabeled regularization
            ru = rampup_factor(iter_num, args.u_rampup_iters)
            L_u_duet = ru * L_u_duet

            # total for both optimizers (stable)
            loss_enc = lb_loss_enc + 0.5 * L_u_duet
            loss_dec = lb_loss_dec + 0.5 * L_u_duet

            # -----------------------------
            # (3) Build inference model and Uncertainty-Aware Pseudo-Label Refinement (CutMix/CPS)
            # -----------------------------
            with torch.no_grad():
                inference_model.decoder.load_state_dict(encoder_model.decoder.state_dict())
                inference_model.encoder.load_state_dict(decoder_model.encoder.state_dict())

                inf_logits, _ = inference_model(unimg)
                inf_plab, inf_conf, inf_probs = get_plab_conf_probs(inf_logits)

                # Uncertainty-aware confidence map from entropy (no agreement masks)
                conf_map = confidence_from_entropy(inf_probs, kappa=args.pl_kappa)  # (B,D,H,W)

                # Hard refinement of each branch pseudo-label:
                # If conf is low, replace branch PL with stable PL from inference model.
                enc_logits_ulb, _ = encoder_model(unimg)
                dec_logits_ulb, _ = decoder_model(unimg)

                enc_plab_ulb, _, _ = get_plab_conf_probs(enc_logits_ulb)
                dec_plab_ulb, _, _ = get_plab_conf_probs(dec_logits_ulb)

                low_conf = (conf_map < args.pl_refine_thres)
                enc_plab_ref = torch.where(low_conf, inf_plab, enc_plab_ulb)
                dec_plab_ref = torch.where(low_conf, inf_plab, dec_plab_ulb)

                # Reliability mask for CPS: directly use confidence (soft)
                rel_mask = conf_map  # (B,D,H,W) in (0,1]

                # context mask (pairwise CutMix)
                img_mask, loss_mask = context_mask(unimg, args.mask_ratio)

                # make loss_mask shape-safe (B,D,H,W)
                if torch.is_tensor(loss_mask):
                    loss_mask = loss_mask.to(device=unimg.device, dtype=unimg.dtype)
                else:
                    loss_mask = torch.tensor(loss_mask, device=unimg.device, dtype=unimg.dtype)

                if loss_mask.dim() == 5 and loss_mask.size(1) == 1:
                    loss_mask = loss_mask.squeeze(1)
                elif loss_mask.dim() == 4 and loss_mask.size(0) == 1:
                    loss_mask = loss_mask.expand(unimg.size(0), *loss_mask.shape[1:])
                elif loss_mask.dim() == 3:
                    loss_mask = loss_mask.unsqueeze(0).expand(unimg.size(0), *loss_mask.shape)

            # -----------------------------
            # (4) CPS with FOCAL loss (instead of CE)
            # -----------------------------
            mixed_img = mix_images_pairs(unimg, img_mask)
            mixed_plab_enc = mix_labels_pairs(enc_plab_ref, img_mask)
            mixed_plab_dec = mix_labels_pairs(dec_plab_ref, img_mask)

            mixed_rel = rel_mask * loss_mask + rel_mask.flip(0) * (1.0 - loss_mask)  # (B,D,H,W)

            mixed_logits_enc, _ = encoder_model(mixed_img)
            mixed_logits_dec, _ = decoder_model(mixed_img)

            # focal maps
            cps_map_enc = focal_map_from_logits(
                mixed_logits_enc, mixed_plab_dec.long(),
                alpha_pos=args.focal_alpha, gamma=args.focal_gamma
            )  # (B,D,H,W)

            cps_map_dec = focal_map_from_logits(
                mixed_logits_dec, mixed_plab_enc.long(),
                alpha_pos=args.focal_alpha, gamma=args.focal_gamma
            )

            cps_enc = (cps_map_enc * mixed_rel).sum() / (mixed_rel.sum() + 1e-6)
            cps_dec = (cps_map_dec * mixed_rel).sum() / (mixed_rel.sum() + 1e-6)
            cps_loss = ru * args.lambda_cps * (cps_enc + cps_dec)

            # -----------------------------
            # (5) Total loss + optimize
            # -----------------------------
            total_loss = loss_enc + loss_dec + cps_loss

            enc_opt.zero_grad()
            dec_opt.zero_grad()
            total_loss.backward()
            enc_opt.step()
            dec_opt.step()

            iter_num += 1

            # EMA updates (unchanged)
            update_ema_variables(encoder_model.encoder, decoder_model.encoder, 0.99)
            update_ema_variables(decoder_model.decoder, encoder_model.decoder, 0.99)

            logging.info(
                f"self iter {iter_num} | "
                f"lb_enc={lb_loss_enc.item():.4f} lb_dec={lb_loss_dec.item():.4f} | "
                f"Uduet={L_u_duet.item():.4f} (LF={L_F.item():.4f}, LP={L_P.item():.4f}) | "
                f"cps={cps_loss.item():.4f} | ru={ru:.3f} | total={total_loss.item():.4f}"
            )

            # -----------------------------
            # (6) Validation / saving (unchanged)
            # -----------------------------
            if iter_num % 200 == 0:
                encoder_model.eval()
                d1 = test_3d_patch.var_all_case_LA(
                    args.root_path, encoder_model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if d1 > best_dice_1:
                    best_dice_1 = round(d1, 4)
                    torch.save(encoder_model.state_dict(), os.path.join(self_snapshot_path, f'iter_{iter_num}_model1_dice_{best_dice_1}.pth'))
                    torch.save(encoder_model.state_dict(), os.path.join(self_snapshot_path, f'{args.model}_best_model1.pth'))
                    logging.info(f"Saved best model1 -> dice={best_dice_1}")
                encoder_model.train()

                decoder_model.eval()
                d2 = test_3d_patch.var_all_case_LA(
                    args.root_path, decoder_model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if d2 > best_dice_2:
                    best_dice_2 = round(d2, 4)
                    torch.save(decoder_model.state_dict(), os.path.join(self_snapshot_path, f'iter_{iter_num}_model2_dice_{best_dice_2}.pth'))
                    torch.save(decoder_model.state_dict(), os.path.join(self_snapshot_path, f'{args.model}_best_model2.pth'))
                    logging.info(f"Saved best model2 -> dice={best_dice_2}")
                decoder_model.train()

                inference_model.decoder.load_state_dict(encoder_model.decoder.state_dict())
                inference_model.encoder.load_state_dict(decoder_model.encoder.state_dict())
                inference_model.eval()
                di = test_3d_patch.var_all_case_LA(
                    args.root_path, inference_model, num_classes=num_classes,
                    patch_size=patch_size, stride_xy=64, stride_z=64
                )
                if di > best_dice_inf:
                    best_dice_inf = round(di, 4)
                    torch.save(inference_model.state_dict(), os.path.join(self_snapshot_path, f'iter_{iter_num}_inference_dice_{best_dice_inf}.pth'))
                    torch.save(inference_model.state_dict(), os.path.join(self_snapshot_path, f'{args.model}_best_model.pth'))
                    logging.info(f"Saved best inference -> dice={best_dice_inf}")

                logging.info(f"self iter {iter_num} | val dice: model1={d1:.4f} model2={d2:.4f} inference={di:.4f}")

            if iter_num >= args.self_max_iteration:
                iterator.close()
                return


# ------------------------- Main -------------------------
if __name__ == "__main__":
    pre_snapshot_path = f"/content/drive/MyDrive/Research/models/DualMatch/{dataset}/ours/{args.exp}_{args.labelnum}_labeled/pre_train"
    self_snapshot_path = f"/content/drive/MyDrive/Research/models/DualMatch/{dataset}/ours/{args.exp}_{args.labelnum}_labeled/self_train"

    print("Starting training.")
    for sp in [pre_snapshot_path, self_snapshot_path]:
        os.makedirs(sp, exist_ok=True)
        if os.path.exists(sp + '/code'):
            shutil.rmtree(sp + '/code')

    try:
        shutil.copy('/content/drive/MyDrive/Research/DuetMatch/train.py', self_snapshot_path)
    except Exception:
        pass

    logging.basicConfig(
        filename=pre_snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if not args.skip_pretrain:
        pre_train(args, pre_snapshot_path)

    self_train(args, pre_snapshot_path, self_snapshot_path)
