from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob
import cv2
import sys
from skimage.transform import resize
from datetime import datetime as dt

import binvox_rw

class CTDataset(Dataset):
    def __init__(self, datapath):
        self.datapath = datapath
        self.datapaths_img = glob.glob(self.datapath + 'image/*')
        self.datapaths_vol = glob.glob(self.datapath + 'volume/*')
        # self.transforms = transforms_

    def __len__(self):
        return len(glob.glob(self.datapath + 'image/*'))

    def __getitem__(self, idx):
        # Get data of rendering images
        image_paths = self.datapaths_img[idx]
        rendering_images = []
        image_path_list = sorted(glob.glob(image_paths + '/rendering/a*.png'))
        for image_path in image_path_list:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            rendering_image = cv2.resize(rendering_image, dsize=(128, 128), interpolation=cv2.INTER_AREA)
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                        (dt.now(), image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)
        rendering_images = np.asarray(rendering_images)
        rendering_images = rendering_images[:, :, :, 0]
        rendering_images = resize(rendering_images, (128, 128, 128))

        # Get data of volume
        volume_path = self.datapaths_vol[idx] + '/model.binvox'
        with open(volume_path, 'rb') as f:
            volume = binvox_rw.read_as_3d_array(f)

        '''
        if self.transforms:
            # image, mask = self.transforms(image), self.transforms(mask)
            image = self.transforms(image)
            '''

        # return {"A": image, "B": mask}
        return {"A": rendering_images, "B": volume.data}
