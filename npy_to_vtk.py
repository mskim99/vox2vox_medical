import vtk
import numpy as np
from vtk.util import numpy_support

data = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/221005_2_log_loss_GAN_vox2vox_normalize_L2_uqi_loss_L1_23_layer_4_LFFB/epoch_200_real_B_16.npy')
'''
data_max = data.max()
data_min = data.min()
data = (data - data_min) / (data_max - data_min)
'''
data = data[:, :, :]
# data = data / 255.
imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([128, 128, 128])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/vox2vox-master/vox2vox-master/logs/221005_2_log_loss_GAN_vox2vox_normalize_L2_uqi_loss_L1_23_layer_4_LFFB/epoch_200_real_B_16.mha')
writer.SetInputData(imdata)
writer.Write()