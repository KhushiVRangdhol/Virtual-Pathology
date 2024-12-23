# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:39:32 2020

"""

import numpy as np

from PIL import Image

import os, sys


# fill up the names of the unstained images


unstained_images = {}


unstained_images[0] = "19AH7970-1.jpg.jpg"
unstained_images[1] = "19AH7973a.jpg.jpg"

unstained_images[2] = '19AH7973a-1.jpg.jpg'

unstained_images[3] = '19AH7970.jpg.jpg'


# fill up the names of the staiend images, should note that corresponding image pair should be given the same number


stained_images = {}

stained_images[0] = '19AH7970 (H_Ex2.4)-1.jpg.jpg'

stained_images[1] = '19AH7973a (H_Ex2.4).jpg.jpg'

stained_images[2] = '19AH7973a (H_Ex2.4)-1.jpg.jpg'

stained_images[3] = '19AH7970 (H_Ex2.4).jpg.jpg'

stained_np_dict = {}

unstained_np_dict = {}

directory="C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\image_pairs\\image_pairs\\"
# Directory that contains stained and unstained images
#directory="C:\\"
unstained_folder= directory+"Unstained_folder\\"
# unstained_folder='C:\\Users\\Desktop\\'

stained_folder= directory+"Stained_folder\\"
#stained_folder='C:\\Users\\Desktop\\'



n1 = len(unstained_images)

n2 = len(stained_images)

if n1 != n2:
   print('Make sure the number of unstained and instained images equal')
   sys.exit()


for k in range(0,n1):

    path = unstained_folder + unstained_images[k]

    img = Image.open(path)

    data = np.array(img, dtype='uint8')

    unstained_np_dict[str(k)] = data


    path = stained_folder + stained_images[k]

    img = Image.open(path)

    data = np.array(img, dtype='uint8')

    stained_np_dict[str(k)] = data


# Output folder location

output_folder = 'C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\image_pairs\\output_image_pairs\\'
#output_folder = 'C:\\Users\\Desktop\\'

output_image1 =  'unstained.npy'

output_image2 = 'stained.npy'


np.save(output_folder+output_image1, unstained_np_dict)

np.save(output_folder+output_image2, stained_np_dict)


print('\nMerging of aligned images completed, plesae check the numpy binary file (.npy) on output folder \n')
