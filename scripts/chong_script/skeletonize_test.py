import numpy as np
import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import torch

downscale = 1.0
# define image parameters
res_height = 480
res_width = 640

# read image and turn it into binary image
img_path = '/home/candice/Documents/Pics/recon_test_data/0000_seg.png'
raw_img_rgb = cv2.imread(img_path)
raw_img_rgb = cv2.resize(raw_img_rgb,
                                (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
raw_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

ret, img_thresh = cv2.threshold(raw_img.copy(), 80, 255, cv2.THRESH_BINARY)
img_thresh = img_thresh

# extend image to produce 
extend_dim = int(60)
img_thresh_extend = np.zeros((res_height, res_width + extend_dim))
img_thresh_extend[0:res_height, 0:res_width] = img_thresh.copy() / 255

left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, res_width - 1]))
left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, res_width - 10]))

extend_vec_pt1_center = np.array([res_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1]) / 2])
extend_vec_pt2_center = np.array(
    [res_width - 5, (left_boundarylineB_id[0] + left_boundarylineB_id[-1]) / 2])
exten_vec = extend_vec_pt2_center - extend_vec_pt1_center

if exten_vec[1]==0:
    exten_vec[1] += 0.00000001

k_extend = exten_vec[0] / exten_vec[1]
b_extend_up = res_width - k_extend * left_boundarylineA_id[0]
b_extend_dw = res_width - k_extend * left_boundarylineA_id[-1]

extend_ROI = np.array([
    np.array([res_width, left_boundarylineA_id[0]]),
    np.array([res_width, left_boundarylineA_id[-1]]),
    np.array([res_width + extend_dim, int(((res_width + extend_dim) - b_extend_dw) / k_extend)]),
    np.array([res_width + extend_dim,
                int(((res_width + extend_dim) - b_extend_up) / k_extend)])
])

img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

skeleton = skeletonize(img_thresh_extend)

img_raw_skeleton = np.argwhere(skeleton[:, 0:res_width] == 1)

# Preprocess the skeleton to fit the pic
centerline_from_img = torch.as_tensor(img_raw_skeleton).float()
if centerline_from_img[0, 1] >= 620:
    centerline_from_img = torch.flip(centerline_from_img, dims=[0])
centerline_from_img = torch.flip(centerline_from_img, dims=[1])

# Draw centerline from skeleton on the image
for i in range(centerline_from_img.shape[0] - 1):
    p1 = (int(centerline_from_img[i, 0]), int(centerline_from_img[i, 1]))
    p2 = (int(centerline_from_img[i + 1, 0]), int(centerline_from_img[i + 1, 1]))
    print(p1)
    cv2.line(raw_img_rgb, p1, p2, (0, 255, 0), 4)
    cv2.circle(raw_img_rgb, p1, radius=1, color=(0,0,255), thickness=-1)

plt.figure()
plt.imshow(raw_img_rgb)
plt.show()
