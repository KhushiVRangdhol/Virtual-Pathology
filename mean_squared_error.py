# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:58:40 2023

@author: khush
"""

import cv2
import numpy as np

source_direc = "C:\\Users\\khush\\OneDrive - City University of Hong Kong - Student\\Courses\\Yr 2 Sem A\\PhysicsinMedicine\\high_resolution_image_pairs\\original_images\\"
# Load two images
img1 = cv2.imread(source_direc+'virtual_03-13170.png')
img2 = cv2.imread(source_direc+'physical_03-13170.png')

#print("Image shape" + image1.shape[0] + ", " + image1.shape[1])
print("Image shape" + str(img1.shape))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


#print("Image shape" + str(img1.shape))
# Ensure the images have the same dimensions
#if image1.shape != image2.shape:
 #   raise ValueError("Images must have the same dimensions for MSE calculation.")

# Calculate the Mean Squared Error
#image_mse = n((image1 **2) - (image2 ** 2))



def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

image_mse = mse(img1, img2)
print(f"Mean Squared Error: {image_mse}")
