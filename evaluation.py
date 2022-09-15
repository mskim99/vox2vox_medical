import torch
import numpy as np

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
test_iou = dict()

for i in range (0, 18):
    fake_B = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/220915_5_log_loss_GAN_prop_normalize_add_mid_layer_4_2048/epoch_200_fake_B_' + str(i).zfill(2) + '.npy')
    real_B = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/220915_5_log_loss_GAN_prop_normalize_add_mid_layer_4_2048/epoch_200_real_B_' + str(i).zfill(2) + '.npy')
    real_B_max = real_B.max()
    real_B_min = real_B.min()
    real_B = (real_B - real_B_min) / (real_B_max - real_B_min)

    fake_B_tensor = torch.Tensor(fake_B)
    real_B_tensor = torch.Tensor(real_B)

    L1_loss = l1_loss(fake_B_tensor, real_B_tensor)
    L1_losses.update(L1_loss.item())

    # IoU per sample
    sample_iou = []
    sample_accuracy = []
    sample_precision = []
    sample_recall = []
    sample_f1_score = []
    for th in [.45, .475, .5, .525, .55]:
        _volume = torch.ge(fake_B_tensor, th).float()
        _gt_volume = torch.ge(real_B_tensor, th).float()

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

    # IoU per taxonomy
    test_iou = {'n_samples': 0, 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    test_iou['n_samples'] += 1
    test_iou['iou'].append(sample_iou)
    test_iou['accuracy'].append(sample_accuracy)
    test_iou['precision'].append(sample_precision)
    test_iou['recall'].append(sample_recall)
    test_iou['f1-score'].append(sample_f1_score)

    print('[INFO] Sample [%d] L1Loss = %.6f' % (i + 1,  L1_loss.item()))

# Print header
print('[INFO] L1Loss Average = %.6f' % L1_losses.avg)
eval_factors = ['iou', 'accuracy', 'precision', 'recall', 'f1-score']
print('============================ RESULTS ============================')
print('#Sample', end='\t')
for th in [.45, .475, .5, .525, .55]:
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
