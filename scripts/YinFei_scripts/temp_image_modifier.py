import cv2
import numpy as np

tumor_img_path = '/home/inffzy/Desktop/ARCLab/ARCLab-CCCatheter/data/contour_images/tumor4595_new.png' 
tumor_img_path_new = '/home/inffzy/Desktop/ARCLab/ARCLab-CCCatheter/data/contour_images/tumor4595_resized.png'


img = cv2.imread(tumor_img_path)
print(img.shape)

img_new = cv2.resize(img, (640, 480))

cv2.imwrite(tumor_img_path_new, img_new)