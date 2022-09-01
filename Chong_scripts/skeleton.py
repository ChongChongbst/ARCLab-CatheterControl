import numpy as np
import cv2
import os
import torch
from skimage.morphology import skeletonize

class Skeleton():
    def __init__(self, img_dir_path, i, downscale=1):
        '''
        find the centerline (skeleton) of one single image
        
        Args:
            img_dir_path: the path that the image is stored
            i: the rank of image in the directory

        Outputs:
            self.skeleton: the skeleton on 2d image extracted
        '''
        self.img_dir_path = img_dir_path
        self.i = i
        self.downscale = downscale
        self.res_height = 480
        self.res_width = 640

        self.process_skeleton()

    def getContourSamples(self, raw_img):
            '''
            Find the contour of the image of the catheter

            Inputs:
                image path is defined in the main function

            Output:
                img_raw_skeleton (numpy array): the skeleton of the catheter on the raw image

            '''

            # binarilize the skeleton image
            ret, img_thresh = cv2.threshold(raw_img.copy(), 80, 255, cv2.THRESH_BINARY)

            # perform skeletonization
            # extend the boundary of the image
            extend_dim = int(60)
            img_thresh_extend = np.zeros((self.res_height, self.res_width + extend_dim))
            img_thresh_extend[0:self.res_height, 0:self.res_width] = img_thresh.copy()/255

            left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, self.res_width -1]))
            left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, self.res_width - 10]))

            extend_vec_pt1_center = np.array([self.res_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1])/2])
            extend_vec_pt2_center = np.array([self.res_width-5,(left_boundarylineB_id[0] + left_boundarylineB_id[-1])/2])
            extend_vec = extend_vec_pt2_center - extend_vec_pt1_center

            if extend_vec[1] == 0:
                extend_vec[1] += 1e-8

            k_extend = extend_vec[0] / extend_vec[1]
            b_extend_up = self.res_width - k_extend * left_boundarylineA_id[0]
            b_extend_dw = self.res_width - k_extend * left_boundarylineA_id[-1]
            
            # get the intersection point with boundary
            extend_ROI  = np.array([
                np.array([self.res_width, left_boundarylineA_id[0]]),
                np.array([self.res_width, left_boundarylineA_id[-1]]),
                np.array([self.res_width + extend_dim, int(((self.res_width + extend_dim) - b_extend_dw) / k_extend)]),
                np.array([self.res_width + extend_dim,
                        int(((self.res_width + extend_dim) - b_extend_up) / k_extend)])        
            ])

            img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

            skeleton = skeletonize(img_thresh_extend)

            img_raw_skeleton = np.argwhere(skeleton[:, 0:self.res_width] == 1)

            return img_raw_skeleton


    def get_centerline(self):
        '''
        Inputs:
            i (int): the ranking of the image in the directory 0-20

        Outputs:
            img_raw_skeleton: the skeleton of the raw image
            
        '''

        i = self.i

        img_path = os.path.join(self.img_dir_path, str(i).zfill(3) + '.png')

        raw_img_rgb = cv2.imread(img_path)

        raw_img_rgb = cv2.resize(raw_img_rgb, (int(raw_img_rgb.shape[1] / self.downscale), int(raw_img_rgb.shape[0] / self.downscale)))
        raw_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

        img_raw_skeleton = self.getContourSamples(raw_img)

        return img_raw_skeleton


    def process_skeleton(self):
        '''
          Arrange the points got directly from the image

        Variables:
            self.img_raw_skeleton: the skeleton points extracted directly from the image taken 
                                    from the projection of the targetted curve
        Results:
            self.skeleton: the arranged skeleton points        
        '''
        img_raw_skeleton = self.get_centerline()
        skeleton = torch.as_tensor(img_raw_skeleton).float()
        if skeleton[0,1] >= 620:
            skeleton = torch.flip(skeleton, dims=[0])
        self.skeleton = torch.flip(skeleton, dims=[1]) 