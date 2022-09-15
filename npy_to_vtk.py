import vtk
import numpy as np
from vtk.util import numpy_support

data = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/220915_5_log_loss_GAN_prop_normalize_add_mid_layer_4_2048/epoch_200_fake_B_01.npy')
'''
data_max = data.max()
data_min = data.min()
data = (data - data_min) / (data_max - data_min)
'''
data = data[0, 0, :, :, :]
# data = data / 255.
imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([128, 128, 128])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/vox2vox-master/vox2vox-master/logs/220915_5_log_loss_GAN_prop_normalize_add_mid_layer_4_2048/epoch_200_fake_B_01.mha')
writer.SetInputData(imdata)
writer.Write()