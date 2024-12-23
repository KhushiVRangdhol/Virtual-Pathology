from __future__ import print_function
import argparse
import os
from math import log10
from PIL import Image
import cv2
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
# from data import get_training_set, get_test_set
import dataloader_npy

start_time = time.time()

gray_transformer = transforms.Normalize(mean=[0.5],
                                    std=[1.0])

def gray_pil2tensor(image, dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : 
        a = np.expand_dims(a,0)
    return torch.from_numpy(a.astype(dtype, copy=False))


# Pretrained_dir places the nueral network model in pth format

pretrained_dir = 'C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\source_code\\source_code\\checkpoint\\256\\'

pretrained_filename = 'netG_model_epoch_200.pth'


# pretrained_dir = '/media/cz/Data/Code/PhaseStain/pix2pix-pytorch/checkpoint/256_hpc/'
# pretrained_filename = 'netG_model_epoch_200.pth'



#model_G = torch.load(pretrained_dir + pretrained_filename).cuda()
model_G = torch.load(pretrained_dir + pretrained_filename, map_location='cpu')
model_G.eval()
#print(model_G)


# Directory that contains unpredicted unstain images and predicted stain images

source_dir = 'C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\image_pairs\\image_pairs\\Gray_scale\\'


source_file = 'Gray_19AH7973a.jpg.jpg'

target_dir = 'C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\image_pairs\\output\\'


# source_dir = '/media/cz/Data/Code/PhaseStain/cz_implement/registered_dataset/masked_registered_patch_dataset/256/each_pair'
# target_dir = '/media/cz/Data/Code/PhaseStain/pix2pix-pytorch/predict/256_hpc_twice/'


img = cv2.imread(os.path.join(source_dir, source_file), cv2.IMREAD_GRAYSCALE)

img_tensor = torch.unsqueeze(gray_transformer(gray_pil2tensor(img/255.0, np.float32)), dim=0)

predict_output = model_G(img_tensor)

predict_img = (predict_output[0,:,:,:]+0.5)*255.0

cv2.imwrite(os.path.join(target_dir, 'predict_'+source_file), np.transpose(predict_img.detach(), (1,2,0)).numpy())

                

print('total time:\t', time.time()- start_time)