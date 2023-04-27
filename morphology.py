import glob
import cv2
import numpy as np

img_list = glob.glob('/scratch/yy3u19/PAIP-2019-master/data/train/*')+glob.glob('/scratch/yy3u19/PAIP-2019-master/data/test/*')+glob.glob('/scratch/yy3u19/PAIP-2019-master/data/val/*')
k = np.ones((3,3) , np.uint8)
for img_name in img_list:
	img = cv2.imread(img_name, flags=-1)
	mask = img[:,:,3]
	mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE , k , iterations = 5)
	img[:,:,3] = mask
	cv2.imwrite(img_name, img)
	print('saving', img_name)
