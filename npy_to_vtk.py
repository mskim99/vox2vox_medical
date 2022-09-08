import vtk
import numpy as np
from vtk.util import numpy_support

data = np.load('J:/Program/vox2vox-master/vox2vox-master/images/220829_3/epoch_200_fake_B_00.npy')
data = data[0, 0, :, :, :]
data = data / 255.
imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([128, 128, 128])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/vox2vox-master/vox2vox-master/images/220829_3/epoch_200_fake_B_00.mha')
writer.SetInputData(imdata)
writer.Write()