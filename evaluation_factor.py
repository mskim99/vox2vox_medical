import torch

from IQA_pytorch import SSIM
import sklearn.metrics as skl_metrics
import skimage.metrics as ski_metrics
# from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import psnr, ssim, uqi, ergas, scc, rase, sam, msssim, vifp
import piq
import numpy as np

def loss_gtv(gv, gtv, thres, mv, b=1):

    # Set Weight
    weight = torch.ones(gtv.shape).cuda()
    weight_more = torch.where(gtv[:, :, :] > thres)
    weight_less = torch.where((gtv[:, :, :] < thres) & (gtv[:, :, :] > thres - 0.05))
    weight[weight_more] = mv
    weight[weight_less] = -10.0 * gtv[weight_less] + (10 * thres + 0.5)

    smooth_l1_loss = torch.nn.SmoothL1Loss(reduce=None, reduction='none', beta=b)

    loss_value = smooth_l1_loss(gv, gtv)
    loss_value = weight * loss_value
    loss_value = torch.mean(loss_value)

    return loss_value

def ls_loss(gv, gtv, thres, ep):

    mahv_gv = mahf(gv - 0.5, ep)
    mahv_gv = mahv_gv.cuda()

    # inside_pos = torch.where(gtv[:] >= thres)
    # outside_pos = torch.where(gtv[:] < thres)

    ai_in = cal_bgtm_in(mahv_gv, gtv)
    ai_out = cal_bgtm_out(mahv_gv, gtv)

    gc1 = gtv - ai_in
    gc1 = gc1 * gc1

    gc2 = gtv - ai_out
    gc2 = gc2 * gc2

    gc = gc1 * mahv_gv + gc2 * (1 - mahv_gv)
    loss_value = gc.sum()
    return loss_value


def dice_factor(gv, gtv, thres, smooth = 1e-5):
    gv_ref_less = torch.le(gv, thres).float()
    gv_ref_more = torch.ge(gv, thres).float()
    gtv_ref_less = torch.le(gtv, thres).float()
    gtv_ref_more = torch.ge(gtv, thres).float()
    FP_loss_intersection = torch.sum(gv_ref_less.mul(gtv_ref_more))
    FP_loss_union = torch.sum(gv_ref_less.add(gtv_ref_more))
    TN_loss_intersection = torch.sum(gv_ref_more.mul(gtv_ref_less))
    TN_loss_union = torch.sum(gv_ref_more.add(gtv_ref_less))
    dice_value = FP_loss_intersection / (FP_loss_union + smooth) + TN_loss_intersection / (TN_loss_union + smooth)
    return dice_value


def dice_factor2(gv, gtv, thres, smooth = 1e-5):
    gv_ref_more = torch.ge(gv, thres).float()
    gtv_ref_more = torch.ge(gtv, thres).float()
    gv_ref_more_sum = torch.sum(gv_ref_more)
    gtv_ref_more_sum = torch.sum(gtv_ref_more)
    intersection = torch.sum(gv_ref_more.mul(gtv_ref_more))
    dice_value = 2. * intersection / (gv_ref_more_sum + gtv_ref_more_sum)
    return dice_value


def ssim_volume(gv, gtv, normalize=True):

    loss_total = 0.0
    for i in range (0, gv.shape[2]):
        gv_part = gv[0, 0, :, :, i]
        gtv_part = gtv[0, 0, :, :, i]
        ssim_loss_part = ski_metrics.structural_similarity(gv_part, gtv_part, vmin=0, vmax=1)
        loss_total += ssim_loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def ssim_volume2(gv, gtv, normalize=True):

    loss_total = 0.0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[0, 0, :, :, i] * 255.).astype(int)
        gtv_part = (gtv[0, 0, :, :, i] * 255.).astype(int)
        ssim_loss_part = ssim(gv_part, gtv_part)
        loss_total += np.mean(ssim_loss_part)

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def uqi_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[0, 0, :, :, i] * 255.).astype(int)
        gtv_part = (gtv[0, 0, :, :, i] * 255.).astype(int)
        loss_part = uqi(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def msssim_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gtv_part = (gtv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        gtv_part = torch.clamp(gtv_part, min=0.0, max=1.0)
        loss_part = piq.multi_scale_ssim(gv_part, gtv_part, kernel_size=7)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def iwssim_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gtv_part = (gtv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        gtv_part = torch.clamp(gtv_part, min=0.0, max=1.0)
        loss_part = piq.information_weighted_ssim(gv_part, gtv_part, kernel_size=7)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def gmsd_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gtv_part = (gtv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        gtv_part = torch.clamp(gtv_part, min=0.0, max=1.0)
        loss_part = piq.gmsd(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def ms_gmsd_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gtv_part = (gtv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        gtv_part = torch.clamp(gtv_part, min=0.0, max=1.0)
        loss_part = piq.multi_scale_gmsd(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def vsi_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gtv_part = (gtv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        gtv_part = torch.clamp(gtv_part, min=0.0, max=1.0)
        loss_part = piq.vsi(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def haarpsi_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gtv_part = (gtv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        gtv_part = torch.clamp(gtv_part, min=0.0, max=1.0)
        loss_part = piq.haarpsi(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def mdsi_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gtv_part = (gtv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        gtv_part = torch.clamp(gtv_part, min=0.0, max=1.0)
        loss_part = piq.mdsi(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def TV_volume(gv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        loss_part = piq.total_variation(gv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total

def brisque_volume(gv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, :, i])
        gv_part = torch.clamp(gv_part, min=0.0, max=1.0)
        loss_part = piq.brisque(gv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


def ergas_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[0, 0, :, :, i] * 255.).astype(int)
        gtv_part = (gtv[0, 0, :, :, i] * 255.).astype(int)
        loss_part = ergas(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def scc_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[0, 0, :, :, i] * 255.).astype(int)
        gtv_part = (gtv[0, 0, :, :, i] * 255.).astype(int)
        loss_part = scc(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def rase_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[0, 0, :, :, i] * 255.).astype(int)
        gtv_part = (gtv[0, 0, :, :, i] * 255.).astype(int)
        loss_part = rase(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def sam_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[0, 0, :, :, i] * 255.).astype(int)
        gtv_part = (gtv[0, 0, :, :, i] * 255.).astype(int)
        loss_part = sam(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def vifp_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[0, 0, :, :, i] * 255.).astype(int)
        gtv_part = (gtv[0, 0, :, :, i] * 255.).astype(int)
        loss_part = vifp(gv_part, gtv_part)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]
    return loss_total


def confusion_score_volume(gv, gtv, thres, normalize=True, type='None'):

    score_total = 0.0
    gv_ref_more = torch.ge(gv, thres).float()
    gtv_ref_more = torch.ge(gtv, thres).float()
    for i in range (0, gv.shape[2]):
        gv_part = gv_ref_more[0, 0,:, :, i]
        gtv_part = gtv_ref_more[0, 0, :, :, i]

        score_part = 0.0
        if type == 'jaccard':
            score_part = skl_metrics.jaccard_score(gtv_part, gv_part, average='micro')
        if type == 'accuracy':
            score_part = skl_metrics.accuracy_score(gtv_part, gv_part)
        if type == 'precision':
            score_part = skl_metrics.precision_score(gtv_part, gv_part, average='micro')
        if type == 'recall':
            score_part = skl_metrics.recall_score(gtv_part, gv_part, average='micro')
        if type == 'f1-score':
            score_part = skl_metrics.f1_score(gtv_part, gv_part, average='micro')

        score_total += score_part

    if normalize:
        score_total = score_total / gv.shape[2]

    return score_total


def psnr_volume(gv, gtv, normalize=True):

    psnr_value_total = 0.0
    for i in range (0, gv.shape[2]):
        gv_part = gv[:, :, i]
        gtv_part = gtv[:, :, i]
        psnr_value_part = ski_metrics.peak_signal_noise_ratio(gv_part, gtv_part)
        psnr_value_total += psnr_value_part

    if normalize:
        psnr_value_total = psnr_value_total / gv.shape[2]
    return psnr_value_total


def AUC_factor(gv, gtv, thres, normalize=True):
    AUC_value_total = 0.0
    gv_ref_more = torch.ge(gv, thres).float()
    gtv_ref_more = torch.ge(gtv, thres).float()
    for i in range (0, gv.shape[2]):
        gv_part = gv_ref_more[0, 0, :, :, i]
        gtv_part = gtv_ref_more[0, 0, :, :, i]
        if torch.equal(gv_part, gtv_part):
            AUC = 0.5
        else:
            AUC = skl_metrics.roc_auc_score(gv_part, gtv_part)
        AUC_value_total += AUC

    if normalize:
        AUC_value_total = AUC_value_total / gv.shape[2]
    return AUC_value_total


def cal_bgtm_in(mahv_gv, gtv):
    im_data = torch.zeros(gtv.shape).cuda()
    im_data = gtv * mahv_gv
    im_data = im_data / mahv_gv
    return im_data


def cal_bgtm_out(mahv_gv, gtv):
    im_data = torch.zeros(gtv.shape).cuda()
    im_data = gtv * (1. - mahv_gv)
    im_data = im_data / (1. - mahv_gv)
    return im_data


def mahf(z, ep):
    im_data = torch.zeros(z.shape).cuda()
    im_data[:, :, :] = z[:, :, :] / ep
    im_data = torch.tanh(im_data)
    im_data = 0.5 * (im_data + 1.)
    return im_data

'''
test_gtv = torch.tensor([0.4, 0.5, 0.6, 0.4])
test_gv = torch.tensor([0.4, 0.5, 0.6, 0.4])

ls_loss_value = 100. * ls_loss(test_gv, test_gtv, 0.5, 1.)

print(ls_loss_value.item())
'''