import numpy as np
import binvox_rw

data = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/220910_2_log_loss_GAN_L1_IoU_loss/epoch_200_real_B_07.npy')
'''
volume_path = 'J:/Program/vox2vox-master/vox2vox-master/data/test/volume/f_0000009/model.binvox'
with open(volume_path, 'rb') as f:
    volume = binvox_rw.read_as_3d_array(f)
data = volume.data
'''
print(data.min())
print(data.max())