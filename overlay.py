# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
# load the image
image = cv2.imread("img.jpg")
mask = cv2.imread('mask.jpg')
mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
cv2.imwrite('mask_1.jpg',mask)
mask = cv2.imread('mask_1.jpg')
alpha=0.5
cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)
cv2.imwrite('output.jpg',image)