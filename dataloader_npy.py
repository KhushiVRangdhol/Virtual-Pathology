# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import os
import torch
# import h5py
import numpy as np
import random
# import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from scipy.ndimage.interpolation import map_coordinates
# from scipy.ndimage.filters import gaussian_filter

def gray_pil2tensor(image, dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : 
        a = np.expand_dims(a,0)
    return torch.from_numpy(a.astype(dtype, copy=False))


def pil2tensor(image, dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : 
        a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))

class AutoStain_Dataset_npy(Dataset):
  
    " Modify the folder to the current directory that places the numpy files "
    
    def __init__(self, stage, folder='C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\image_pairs\\output_image_pairs', size=256, augmentations=True, mean=[0.5, 0.5, 0.5],std=[1.0, 1.0, 1.0]):        
    #def __init__(self, stage, folder='/media/cz/Data/Data/stain/Patch_dataset_npy/', size=256, augmentations=True, mean=[0.5, 0.5, 0.5],std=[1.0, 1.0, 1.0]):
        """
        random_crop=False,
        random_fliplr=True, random_flipud=True, random_rotation=True,
        color_jitter=False, brightness=0.1
        """
        super(AutoStain_Dataset_npy, self).__init__()
        # basic settings
        self.folder = folder
        self.size = size
        self.stage = stage
        self.augmentations = augmentations

        # color augmentation
        self.RANDOM_BRIGHTNESS = 7
        self.RANDOM_CONTRAST = 5
        self.mean = mean
        self.std = std
        self.gray_transformer = transforms.Normalize(mean=[self.mean[0]],
                                    std=[self.std[0]])
        self.transformer = transforms.Normalize(mean=self.mean,
                                    std=self.std)
        # dataset load
        #self._raw_inputs = (np.load(os.path.join(folder, str(self.size)+'/'+self.stage+'/'+str(self.size)+'_'+self.stage+'_gray.npy'),encoding="latin1", allow_pickle=True)).item(0)
        
        #self._raw_targets = (np.load(os.path.join(folder, str(self.size)+'/'+self.stage+'/'+str(self.size)+'_'+self.stage+'_he.npy'),encoding="latin1", allow_pickle=True)).item(0)
        
        " Modify the Numpy files names if necessary "
        
        self._raw_inputs = (np.load(os.path.join(folder, 'unstained.npy'),encoding="latin1", allow_pickle=True)).item(0)
        
        self._raw_targets = (np.load(os.path.join(folder, 'stained.npy'),encoding="latin1", allow_pickle=True)).item(0)
        
              
        print('load {} dataset'.format(self.stage))

    def __getitem__(self, index):
        input_ = self._raw_inputs[str(index)]
        target_ = self._raw_targets[str(index)]
        # im = np.expand_dims(im, axis=2).copy()

        if self.augmentations:
            # random flip
            if random.uniform(0, 1) < 0.5:
                input_ = np.fliplr(input_)
                target_ = np.fliplr(target_)
            if random.uniform(0, 1) < 0.5:
                input_ = np.flipud(input_)
                target_ = np.flipud(target_)

            # random rotation
            r = random.randint(0, 3)
            if r:
                input_ = np.rot90(input_, r)
                target_ = np.rot90(target_, r)
            # cast to float
            input_ = input_.astype(np.float32) / 255.0
            target_ = target_.astype(np.float32) / 255.0
            # color jitter
            br = random.randint(-self.RANDOM_BRIGHTNESS, self.RANDOM_BRIGHTNESS) / 100.
            input_ = input_ + br
            target_ = target_ + br
            # Random contrast
            cr = 1.0 + random.randint(-self.RANDOM_CONTRAST, self.RANDOM_CONTRAST) / 100.
            input_ = input_ * cr
            target_ = target_ * cr
            # clip values to 0-1 range
            input_ = np.clip(input_, 0, 1.0)
            target_ = np.clip(target_, 0, 1.0)
            input_ = self.gray_transformer(gray_pil2tensor(input_, np.float32))
            target_ = self.transformer(pil2tensor(target_, np.float32))

        else:
            input_ = input_.astype(np.float32) / 255.0
            target_ = target_.astype(np.float32) / 255.0
            input_ = self.gray_transformer(gray_pil2tensor(input_, np.float32))
            target_ = self.transformer(pil2tensor(target_, np.float32))
            # [0, 1] -> [-0.5, 0.5]

        return input_, target_


    def __len__(self):
        return len(self._raw_targets)


def get_loader(batch_size, stage, size, num_workers, augmentations, mean=[0.5,0.5,0.5], std=[1,1,1]):

    if stage == 'train':
        dataset = AutoStain_Dataset_npy(stage='train', size=size, augmentations=augmentations, mean=mean, std=std)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        print('build auto stain train dataset with {} num_workers'.format(num_workers))
    elif stage == 'test':
        dataset = AutoStain_Dataset_npy(stage='test', size=size, augmentations=augmentations, mean=mean, std=std)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        print('build auto stain test dataset with {} num_workers'.format(num_workers))
    else:
        print('dataloader stage problem')

    return dataloader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataloader = get_loader(batch_size=1, stage='test', size=256, num_workers=1, augmentations=False, mean=[0.5,0.5,0.5], std=[1,1,1])
    for idx, (input_, target_) in enumerate(dataloader):
        if idx == 2:
            print(target_)
            plt.imshow(input_.numpy()[0,0,:,:]+0.5)
            # plt.imshow(np.transpose(target_.numpy()[0,:,:,:], (1,2,0))+0.5)
            plt.show()
            break
