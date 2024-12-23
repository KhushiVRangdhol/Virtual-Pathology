import numpy as np 
import skimage
import os
from skimage import img_as_float
from skimage import io
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

#import ssim.ssimlib as pyssim

import time

start_time = time.time()

#gt_256_dir = '/media/cz/Data/Code/PhaseStain/cz_implement/registered_dataset/masked_registered_patch_dataset/512/each_pair/'
#predict_256_1st_dir = '/media/cz/Data/Code/PhaseStain/pix2pix-pytorch/predict/512_hpc/'


predict_dir = 'C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\high_resolution_image_pairs\\original_images\\'

predict_image = {} 

predict_image[0] = 'virtual_03-13170.png'
predict_image[1] = 'virtual_06AH21654_1_0.png'
predict_image[2] = 'virtual_12AH3884_2_0.png'
predict_image[3] = 'virtual_12AH14738_1_1.png'
predict_image[4] = 'virtual_B274.png'
predict_image[5] = 'virtual_G253B25_0.png'


original_HE_dir = 'C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\high_resolution_image_pairs\\original_images\\'


original_HE_image = {}


original_HE_image[0] = 'physical_03-13170.png'
original_HE_image[1] = 'physical_06AH21654_1_0.png'
original_HE_image[2] = 'physical_12AH3884_2_0.png'
original_HE_image[3] = 'physical_12AH14738_1_1.png'
original_HE_image[4] = 'physical_B274.png'
original_HE_image[5] = 'physical_G253B25_0.png'


n1 = len(predict_image)

n2 = len(original_HE_image)

if n1 != n2:
   print('Make sure the number of predicted and original H and E images are equal')
   sys.exit() 



for k in range(0,n1):
       
    predict_file = predict_dir + predict_image[k]
    
    original_HE_file = original_HE_dir + original_HE_image[k]
    
    predict_patch = img_as_float(skimage.io.imread(predict_file))
    
    gt_patch = img_as_float(skimage.io.imread(original_HE_file))
    
    psnr_patch = psnr(gt_patch, predict_patch, data_range=predict_patch.max() - predict_patch.min())
  # print(psnr_patch)
  
    print('\nImage Pair', k+1, ' ',predict_image[k],' vs ',original_HE_image[k],' PSNR =',psnr_patch)


