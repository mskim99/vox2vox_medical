import torch
import numpy as np
import sklearn.metrics as skl_metrics
import warnings
# from scipy.spatial.distance import directed_hausdorff

import evaluation_factor as ef

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

l1_loss = torch.nn.L1Loss()
L1_losses = AverageMeter()
V_SSIMs = AverageMeter()
dice_scores = AverageMeter()
V_PSNRs = AverageMeter()
f1_scores = AverageMeter()
V_uqis = AverageMeter()
V_msssims = AverageMeter()
V_ergass = AverageMeter()
V_sccs = AverageMeter()
V_rases = AverageMeter()
V_sams = AverageMeter()
V_vifps = AverageMeter()

test_iou = dict()
warnings.filterwarnings('ignore')

for i in range (0, 18):
    fake_B = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/220923_7_log_loss_GAN_vox2vox_normalize_L2_dice_loss_33_43_33/epoch_200_fake_B_' + str(i).zfill(2) + '.npy')
    real_B = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/220923_7_log_loss_GAN_vox2vox_normalize_L2_dice_loss_33_43_33/epoch_200_real_B_' + str(i).zfill(2) + '.npy')
    # fake_B = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/211221_3_loss_GAN_test_epoch_400_G_1_D_linear_increase_10_r_img_4_L1_4_SSIM_2_IoU_drp_g_e_0_375_lr_1e-4_norm_res_128/gv_' + str(i).zfill(6) + '.npy')
    # real_B = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/211221_3_loss_GAN_test_epoch_400_G_1_D_linear_increase_10_r_img_4_L1_4_SSIM_2_IoU_drp_g_e_0_375_lr_1e-4_norm_res_128/gtv_' + str(i).zfill(6) + '.npy')
    '''
    real_B_max = real_B.max()
    real_B_min = real_B.min()
    real_B = (real_B - real_B_min) / (real_B_max - real_B_min)
    '''
    fake_B_tensor = torch.Tensor(fake_B)
    real_B_tensor = torch.Tensor(real_B)
    '''
    fake_B = fake_B.reshape((1, 1) + fake_B.shape)
    real_B = real_B.reshape((1, 1) + real_B.shape)
    fake_B_tensor = fake_B_tensor.unsqueeze(0)
    fake_B_tensor = fake_B_tensor.unsqueeze(1)
    real_B_tensor = real_B_tensor.unsqueeze(0)
    real_B_tensor = real_B_tensor.unsqueeze(1)
    '''
    L1_loss = l1_loss(fake_B_tensor, real_B_tensor)
    L1_losses.update(L1_loss.item())

    V_SSIM = ef.ssim_volume2(fake_B, real_B, normalize=True)
    V_SSIMs.update(V_SSIM)

    V_PSNR = ef.psnr_volume(fake_B, real_B, normalize=True)
    V_PSNRs.update(V_PSNR)

    V_uqi = ef.uqi_volume(fake_B, real_B, normalize=True)
    V_uqis.update(V_uqi)

    V_msssim = ef.msssim_volume(fake_B, real_B, normalize=True)
    V_msssims.update(V_msssim)

    V_ergas = ef.ergas_volume(fake_B, real_B, normalize=True)
    V_ergass.update(V_ergas)

    V_scc = ef.scc_volume(fake_B, real_B, normalize=True)
    V_sccs.update(V_scc)

    V_rase = ef.rase_volume(fake_B, real_B, normalize=True)
    V_rases.update(V_rase)

    V_sam = ef.sam_volume(fake_B, real_B, normalize=True)
    V_sams.update(V_sam)

    V_vifp = ef.vifp_volume(fake_B, real_B, normalize=True)
    V_vifps.update(V_vifp)

    # h_dist = max(directed_hausdorff(fake_B, real_B)[0], directed_hausdorff(real_B, fake_B)[0])
    # h_dists.update(h_dist)

    dice_score = 0.0
    AUC_score = 0.0
    f1_score = 0.0
    thres_list = np.arange(0.3, 1.0, 0.01)
    for thres in thres_list:
        dice_score_part = ef.dice_factor2(fake_B_tensor, real_B_tensor, thres)
        dice_score += dice_score_part
        # f1_score_part = ef.f1_score_volume(fake_B_tensor, real_B_tensor, thres)
        # f1_score += f1_score_part
    dice_scores.update(dice_score)
    # f1_scores.update(f1_score)
    # AUC_scores.update(AUC_score)
    '''
    fake_B = fake_B[0, 0, :, :, :]
    real_B = real_B[0, 0, :, :, :]
    fake_B_tensor = fake_B_tensor.squeeze(0)
    fake_B_tensor = fake_B_tensor.squeeze(1)
    real_B_tensor = real_B_tensor.squeeze(0)
    real_B_tensor = real_B_tensor.squeeze(1)
    '''
    # IoU per sample
    sample_iou = []
    sample_accuracy = []
    sample_precision = []
    sample_recall = []
    sample_f1_score = []
    for th in [.25, .2875, .325, .3625, .4, .45, .475, .5, .525, .55]:
        _volume = torch.ge(fake_B_tensor, th).float()
        _gt_volume = torch.ge(real_B_tensor, th).float()

        iou = ef.confusion_score_volume(real_B_tensor, fake_B_tensor, th, normalize=True, type='jaccard')
        accuracy = ef.confusion_score_volume(real_B_tensor, fake_B_tensor, th, normalize=True, type='accuracy')
        precision = ef.confusion_score_volume(real_B_tensor, fake_B_tensor, th, normalize=True, type='precision')
        recall = ef.confusion_score_volume(real_B_tensor, fake_B_tensor, th, normalize=True, type='recall')
        f1_score = ef.confusion_score_volume(real_B_tensor, fake_B_tensor, th, normalize=True, type='f1-score')

        '''
        volume_num = torch.sum(_volume).float()
        gt_volume_num = torch.sum(_gt_volume).float()
        total_voxels = float(128 * 128 * 128)

        intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
        union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()

        iou = intersection / union

        TP = intersection / total_voxels
        TN = 1.0 - (union / total_voxels)
        FP = (volume_num - intersection) / total_voxels
        FN = (gt_volume_num - intersection) / total_voxels

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        sample_iou.append(iou.item())
        sample_accuracy.append(((TP + TN) / (TP + TN + FP + FN)).item())
        sample_precision.append(precision.item())
        sample_recall.append(recall.item())
        sample_f1_score.append(((2. * precision * recall) / (precision + recall)).item())
        '''

        sample_iou.append(iou.item())
        sample_accuracy.append(accuracy.item())
        sample_precision.append(precision.item())
        sample_recall.append(recall.item())
        sample_f1_score.append(f1_score.item())

    # IoU per taxonomy
    test_iou = {'n_samples': 0, 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    test_iou['n_samples'] += 1
    test_iou['iou'].append(sample_iou)
    test_iou['accuracy'].append(sample_accuracy)
    test_iou['precision'].append(sample_precision)
    test_iou['recall'].append(sample_recall)
    test_iou['f1-score'].append(sample_f1_score)

    print('[INFO] Sample [%d] L1 Error = %.6f, Dice Score = %.6f, PSNR score =  %.6f, SSIM score =  %.6f, UQI score =  %.6f, '
          'MS-SSIM score =  %.6f, ERGAS score =  %.6f, SCC score =  %.6f, RASE score =  %.6f, SAM score =  %.6f, VIF score =  %.6f'
          % (i + 1,  L1_loss.item(), dice_score, V_PSNR.item(), V_SSIM.item(), V_uqi.item(), V_msssim.item(), V_ergas.item()
             , V_scc.item(), V_rase.item(), V_sam.item(), V_vifp.item()))

# Print header
print('[INFO] Average : L1Loss = %.6f, Dice Scores = %.6f, PSNR score =  %.6f, SSIM score =  %.6f, UQI score =  %.6f, '
      'MS-SSIM score =  %.6f, ERGAS score =  %.6f, SCC score =  %.6f, RASE score =  %.6f, SAM score =  %.6f, VIF score =  %.6f'
      % (L1_losses.avg, dice_scores.avg, V_PSNRs.avg, V_SSIMs.avg, V_uqis.avg, V_msssims.avg, V_ergass.avg, V_sccs.avg
         , V_rases.avg, V_sams.avg, V_vifps.avg))
eval_factors = ['iou', 'accuracy', 'precision', 'recall', 'f1-score']
print('============================ RESULTS ============================')
print('#Sample', end='\t')
for th in [.25, .2875, .325, .3625, .4, .45, .475, .5, .525, .55]:
    print('t=%.2f' % th, end='\t')
print()
# Print body
for eval_factor in eval_factors:
    print('%s' % eval_factor.ljust(8), end='\t')
    print('%d' % test_iou['n_samples'], end='\t')
    for ti in test_iou[eval_factor]:
        for tj in ti:
            print('%.4f' % tj, end='\t')
    print('')
print('\n')
