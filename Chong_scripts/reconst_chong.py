import pstats
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.append('/home/candice/Documents/ARCLab-CCCatheter/scripts')

from skimage.morphology import skeletonize

from find_curvature import FindCurvature

import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec 

import torch 
import shutil
import os
import pdb
import argparse
import imageio
import random

class reconstructCurve():
    def __init__(self, img_path, curve_length_gt, P0_gt, para_gt, para_init, loss_weight, total_itr, verbose=1):
        '''
        Args:
            P0_gt ((1,3) tensor): the start point of the ground truth bezier curve
            para_gt ((1,6) tensor): the middle and end points of the ground truth bezier curve
            para_init ((1,6) tensor): the middle and end points of the constantly evaluated bezier curve
            loss_weight (): the weight of loss in learning process
            verbose (0 or 1): if the program need to print the result or not

        '''
        self.curve_length_gt = curve_length_gt
        self.P0_gt = P0_gt
        self.para_gt = para_gt
        self.para = para_init
        self.loss_weight = loss_weight
        self.total_itr = total_itr 
        self.verbose = verbose
        # the path to save the pictures
        self.path = '/home/candice/Documents/ARCLab-CatheterControl/results'

        self.OFF_SET = torch.tensor([0.00, 0.00, 0.00])

        self.res_width = 640
        self.res_height = 480

        downscale = 1.0

        # camera E parameters
        cam_RT_H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        invert_y = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.cam_RT_H = torch.matmul(invert_y, cam_RT_H)

        # camera I parameters
        self.cam_K = torch.tensor([[883.00220751, 0.0, 320.0], [0.0, 883.00220751, 240.0], [0, 0, 1.0]])
        self.cam_K = self.cam_K / downscale
        self.cam_K[-1, -1] = 1
        
        # Initialize the image
        raw_img_rgb = cv2.imread(img_path)
        self.raw_img_rgb = cv2.resize(raw_img_rgb, (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
        self.raw_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

        self.saved_opt_history = np.zeros((1, self.para.shape[0] + 1))

        # get raw image skeleton
        self.getContourSamples()
        self.processSkeletonImg()

        # get ground truth 3D bezier curve
        self.pos_bezier_3D_gt = self.getBezier3pts(self.para_gt, self.P0_gt)
        self.pos_bezier_3D_init = self.getBezier3pts(para_init, self.P0_gt)


    ##### ===========================================
    #       Image Process Functions Group
    ##### =========================================== 
        
    def getContourSamples(self):
        '''
        Find the contour of the image of the catheter

        Args:
            image path is defined in the main function

        output:
            self.img_raw_skeleton (numpy array): the skeleton of the catheter on the raw image

        '''

        # binarilize the skeleton image
        ret, img_thresh = cv2.threshold(self.raw_img.copy(), 80, 255, cv2.THRESH_BINARY)
        self.img_thresh = img_thresh

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

        self.img_raw_skeleton = np.argwhere(skeleton[:, 0:self.res_width] == 1)

        ref_point1 = None

        return ref_point1
        

    def isPointInImage(self, p_proj, width, height):
        '''
        Determine if a point is within the image or not

        Args:
            p_proj ((2,1) tensor): the point on the 2D frame to be determined if it is in the image or not
            width (int): the width of the image
            height (int): the height of the image
        '''
        if torch.all(torch.isnan(p_proj)):
            # print('NaN')
            return False
        if p_proj[0] < 0 or p_proj[1] < 0 or p_proj[0] > width or p_proj[1] > height:
            # print('Out')
            return False
        return True


    def getProjPointCam(self, p, cam_K):
        '''
        get the 2D point position on the image through 3D points and camera matrix

        Args:
            p ((3,) or (,3) tensor): 3D point in the world frame
            cam_K (tensor): camera matrix
        '''
        if p.shape == (3, ):
            p = torch.unsqueeze(p, dim=0)

        divide_z = torch.div(torch.transpose(p[:, :-1], 0, 1), p[:, -1])

        divide_z = torch.cat((divide_z, torch.ones(1, p.shape[0])), dim=0)

        return torch.transpose(torch.matmul(cam_K, divide_z)[:-1, :], 0, 1)


    ##### ===========================================
    #       Curve Functions Group
    ##### =========================================== 

    def getBezier3pts(self, para, P0, num_samples=200):
        '''
        using the three parameters to find the points on bezier curve
        '''
        sample_list = torch.linspace(0,1,num_samples)

        P1 = P0
        PC = torch.tensor([para[0], para[1], para[2]])
        P2 = torch.tensor([para[3], para[4], para[5]])
        P1p = 2 / 3 * PC + 1 / 3 * P1
        P2p = 2 / 3 * PC + 1 / 3 * P2

        # Get positions and normals from samples along bezier curve
        pos_bezier = torch.zeros(num_samples, 3)
        der_bezier = torch.zeros(num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P1 + 3 * s * (1 - s)**2 * \
                P1p + 3 * (1 - s) * s**2 * P2p + s**3 * P2

        return pos_bezier


    def getBezier4pts(self, control_pts, num_samples=200):
        '''
        Using the control points to find the corresponding number of points on the bezier curve

        Inputs:
            control_pts ((4,3) tensor): the four control points of the bezier curve

        Variables:
            self.num_samples: the number of points should be extracted from the bezier curve
            self.pos_bezier_3D: 3D points on the bezier curve
            self.pos_bezier_cam: 2D points on the projected bezier curve
            self.der_bezier_cam: 2D directions on the projected bezier curve
        '''

        P1 = control_pts[0, :]
        P1p = control_pts[1, :]
        P2 = control_pts[2, :]
        P2p = control_pts[3, :]

        sample_list = torch.linspace(0, 1, num_samples)

        # Get positions and normals from samples along bezier curve
        pos_bezier = torch.zeros(num_samples, 3)
        der_bezier = torch.zeros(num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P1 + 3 * s * (1 - s)**2 * \
                P1p + 3 * (1 - s) * s**2 * P2p + s**3 * P2
            der_bezier[i, :] = -(1 - s)**2 * P1 + ((1 - s)**2 - 2 * s * (1 - s)) * P1p + (-s**2 + 2 *
                                                                                          (1 - s) * s) * P2p + s**2 * P2

        # Convert positions and normals to camera frame
        self.pos_bezier_3D = pos_bezier
        pos_bezier_H = torch.cat((pos_bezier, torch.ones(num_samples, 1)), dim=1)

        pos_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(pos_bezier_H, 0, 1)), 0, 1)
        self.pos_bezier_cam = pos_bezier_cam_H[1:, :-1]

        der_bezier_H = torch.cat((der_bezier, torch.zeros((num_samples, 1))), dim=1)
        der_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[1:, :], 0, 1)), 0,
                                           1)
        self.der_bezier_cam = der_bezier_cam_H[:, :-1]


    def processCenterlineItr(self):
        '''
        Arrage the points got diresctly from bezier curve

        Variables: 
            self.proj_bezier_img: the projected 2D points on image frame of the parameters updated through every iteration
                                from function [getBezier4pts] <-- here the number of points is defined
        Results:
            self.pts_from_bezier: the points projected to 2D image frame updated through every iteration
        '''
        pts_from_bezier = torch.clone(self.proj_bezier_img)
        self.pts_from_bezier = torch.flip(pts_from_bezier, dims=[0])


    def processSkeletonImg(self):
        '''
        Arrange the points got directly from the image

        Variables:
            self.img_raw_skeleton: the skeleton points extracted directly from the image taken 
                                    from the projection of the targetted curve
        Results:
            self.skeleton: the arranged skeleton points       
        '''
        skeleton = torch.as_tensor(self.img_raw_skeleton).float()
        if skeleton[0,1] >= 620:
            skeleton = torch.flip(skeleton, dims=[0])
        self.skeleton = torch.flip(skeleton, dims=[1])        


    def findCorrespCenterlineItr(self):
        '''
        Using the centerline points got directly from the image
        to find the projected centerline of the 3D bezier curve fitted

        Variables:
            self.proj_bezier_img: the projected 2D points on image frame of the parameters updated through every iteration
                                from function [getBezier4pts] <-- in here the number of points is defined
            self.img_raw_skeleton: the centerline extracted from the image taken from monocular camera
        Result:
            self.centerline: the closest point on the projected fitted curve to the skeleton points on the image      
        '''
        skeleton = torch.clone(self.skeleton)
        self.processCenterlineItr()
        pts_from_bezier = torch.clone(self.pts_from_bezier)

        centerline = []
        for i in range(skeleton.shape[0]):
            err = torch.linalg.norm(skeleton[i] - pts_from_bezier, ord=None, axis=1)
            index = torch.argmin(err)
            temp = pts_from_bezier[index, ]
            centerline.append(temp)
        self.centerline = torch.stack(centerline)


    ##### ===========================================
    #       Obj Functions Group
    ##### =========================================== 

    def getCenterlineSegmentsObj(self):
        '''
        get obj from finding centerline points correspondence

        Variables:
            self.centerline: the points from *pts_from_bezier* arranged by *skeleton*

        Results:
            err_skeleton_sum_by_corresp: the error of the difference between desired centerline and the centerline in very iteration
            err_obj_tip: the difference of the tip position
        '''
        self.findCorrespCenterlineItr()

        skeleton = torch.clone(self.skeleton)
        pts_from_bezier = torch.clone(self.pts_from_bezier)

        err_skeleton_by_corresp = torch.linalg.norm(skeleton - self.centerline, ord=None, axis=1) / 1.0
        err_skeleton_sum_by_corresp = torch.sum(err_skeleton_by_corresp) / pts_from_bezier.shape[0]

        err_obj_Tip = torch.linalg.norm(skeleton[0, :] - pts_from_bezier[0, :], ord=None)

        return err_skeleton_sum_by_corresp, err_obj_Tip

    def CurvatureFitted3D(self, n):
        '''
        Find the curvature change of 1000 points on the fitted 3D curve

        Args:
            n (int): the interval of points
        '''
        findk = FindCurvature(self.P0_gt, self.para)
        p3d_fitted_1000 = findk.get_points_on_curve(1000)
        k_fitted_1000 = findk.curvature_change(p3d_fitted_1000,n,1)
        p2d_fitted_1000 = findk.project_3d_to_2d(p3d_fitted_1000) 

        k_fitted_1000 = torch.from_numpy(k_fitted_1000)
        p2d_fitted_1000 = torch.from_numpy(p2d_fitted_1000)
        p2d_fitted_1000 = torch.flip(p2d_fitted_1000, dims=[0])  

        # the value is updated for the plot function
        self.p2d_fitted_1000 = p2d_fitted_1000.clone().detach()

        return k_fitted_1000     


    def getCurvatureObj(self):
        '''
        return the difference of discrete curvature

        Variables:
            k_by_corresp ((N,1) tensor): the curvature change of every iteration
            k ((N,1) tensor): the curvature change of the targeted catheter pose projected on the 2D frame
        '''

        # the skeleton obtained from raw image
        # the final position(2D) that the bezier curve(3D) should fit
        skeleton = torch.as_tensor(self.img_raw_skeleton).float()
        if skeleton[0,1] >= 620:
            skeleton = torch.flip(skeleton, dims=[0])
        skeleton = torch.flip(skeleton, dims=[1])

        N = len(skeleton)
        n = 10
        k = torch.zeros(N-2)
        print(N)

        # Find the curvature change value of the 3D curvature fitted
        # from every iteration
        k_fitted_1000 = self.CurvatureFitted3D(n)

        # find the corresponding curvature on the projected image
        k_by_corresp = []
        # index value for debuging
        indexes = []
        # the first curvature must be corresponding tip
        k_by_corresp.append(k_gt_1000[0])
        indexes.append(torch.tensor(0))

        p2d_gt = []
        k_gt = []

        for i in range(1000):
            if self.isPointInImage(p2d_gt_1000[i], self.res_width,self.res_height):
                p2d_gt.append(p2d_gt_1000[i])
                k_gt.append(k_gt_1000[i])                

        p2d_gt = torch.stack(p2d_gt)
        k_gt = torch.stack(k_gt)

        for i in range(1,N-2):
            err = torch.linalg.norm(skeleton[i] - p2d_gt, dim=1)
            #pdb.set_trace()
            index = torch.argmin(err)
            # index values for debuging
            indexes.append(index)
            temp = k_gt[index, ]
            # in case of finding the same curvature in different indexes and
            # in case of finding the curvature before the searched indexes
            k_by_corresp.append(temp)

        k_by_corresp = torch.stack(k_by_corresp)
        self.indexes = torch.stack(indexes)
        # end of the finding process
        # pdb.set_trace()
        # print(indexes)
        p2d = torch.zeros((3,3))    # store 2d skeleton obtained from image

        A = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float)

        # find curvature change on projected image      
        for i in range(N-1)[1:]:
            if (i-n) < 0:
                p2d[0] = torch.cat((skeleton[0], torch.tensor([1])), 0)
            else:
                p2d[0] = torch.cat((skeleton[i-n], torch.tensor([1])), 0)

            p2d[1] = torch.cat((skeleton[i], torch.tensor([1])), 0)

            if (i+n) > (N-1):
                p2d[2] = torch.cat((skeleton[N-1], torch.tensor([1])), 0)
            else:
                p2d[2] = torch.cat((skeleton[i+n], torch.tensor([1])), 0)

            d1 = torch.linalg.norm(p2d[0]-p2d[1]) + 1e-08
            d2 = torch.linalg.norm(p2d[1]-p2d[2]) + 1e-08
            d3 = torch.linalg.norm(p2d[0]-p2d[2]) + 1e-08

            s = 0.5 * torch.matmul((p2d[0]-p2d[2]), torch.matmul(A, torch.transpose((p2d[0]-p2d[1]), -1, 0)))

            k[i-1] = 4 * s / (d1 * d2 * d3)

        diff_k = torch.subtract(k, k_by_corresp)
        obj_J_k = torch.nansum(diff_k)/N

        # save the plotted figures in folder
        k_plt = k.detach().numpy()

        x1 = range(1,len(k_plt)+1)
        x2 = range(1,N-1)
        k_by_corresp_plt = k_by_corresp.detach().numpy()
        plt.figure()
        plt.plot(x1, k_plt, color='b')
        plt.plot(x2, k_by_corresp_plt, color='r')
        #plt.xlim(1,N-1)
        plt.ylabel('curvature')
        plt.title('the evaluation of curvature')
        plt.savefig(self.path + '/eval/' +str(self.GD_Iteration)+'.jpg')
        plt.close()
        # save the plotted figures in folder       

        #pdb.set_trace()     
        return obj_J_k




    ##### ===========================================
    #       Plot Functions Group
    ##### =========================================== 

    def plotProjCenterline(self):
        centerline_draw_img_rgb = self.raw_img_rgb.copy()
        curve_3D_opt = self.pos_bezier_3D.detach().numpy()
        curve_3D_gt = self.pos_bezier_3D_gt.detach().numpy()
        curve_3D_init = self.pos_bezier_3D_init.detach().numpy()
        error_list = self.error.detach().numpy()
        error = np.linalg.norm(error_list)

        # Draw Centerline
        for i in range(self.proj_bezier_img.shape[0] - 1):
            p1 = (int(self.proj_bezier_img[i, 0]), int(self.proj_bezier_img[i, 1]))
            p2 = (int(self.proj_bezier_img[i + 1, 0]), int(self.proj_bezier_img[i + 1, 1]))
            cv2.line(centerline_draw_img_rgb, p1, p2, (0, 100, 255), 4)

        # Show
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        ax = axes.ravel()
        ax[0].remove()
        ax[1].remove()
        ax[2].remove()
        ax[3].remove()

        gs = GridSpec(2, 2, figure=fig)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(cv2.cvtColor(centerline_draw_img_rgb, cv2.COLOR_BGR2RGB))
        ax0.set_title('Projected centerline')

        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax1.plot3D(curve_3D_gt[:, 0], curve_3D_gt[:, 1], curve_3D_gt[:, 2], color='#1f640a', linestyle='-', linewidth=2)
        ax1.plot3D(curve_3D_init[:, 0],
                   curve_3D_init[:, 1],
                   curve_3D_init[:, 2],
                   color='#a64942',
                   linestyle='--',
                   linewidth=2)  ## green
        ax1.plot3D(curve_3D_opt[:, 0],
                   curve_3D_opt[:, 1],
                   curve_3D_opt[:, 2],
                   color='#6F69AC',
                   linestyle='-',
                   linewidth=2)
        ax1.scatter(curve_3D_opt[-1, 0], curve_3D_opt[-1, 1], curve_3D_opt[-1, 2], marker='^', s=20,
                    c=['#FFC069'])  ## yellow
        ax1.scatter(curve_3D_opt[0, 0], curve_3D_opt[0, 1], curve_3D_opt[0, 2], marker='o', s=20, c=['#FFC069'])

        ax1.quiver([0.0], [0.0], [0.0], [0.02], [0.0], [0.0], length=0.015, normalize=True, colors=['#911F27'])
        ax1.quiver([0.0], [0.0], [0.0], [0.00], [0.005], [0.0], length=0.003, normalize=True, colors=['#57CC99'])
        ax1.quiver([0.0], [0.0], [0.0], [0.00], [0.0], [0.04], length=0.04, normalize=True, colors=['#22577A'])

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.locator_params(nbins=4, axis='x')
        ax1.locator_params(nbins=4, axis='y')
        ax1.locator_params(nbins=4, axis='z')
        ax1.view_init(22, -26)
        ax1.set_title('gt/init/opt : green/red/blue')

        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(self.saved_opt_history[1:, 0], color='#6F69AC', linestyle='-', linewidth=1)
        #ax2.text(-5, 60, 'Error = '+ str(error), fontsize = 15, position=(40,100))
        ax2.set_xlabel('Iterations and error=' + str(error))
        ax2.set_ylabel('Loss')

        # plt.tight_layout()
        plt.show()


    def drawlineandpts(self):
        '''
        draw projected centerline on the image

        Variables:

        '''
        skeleton = torch.as_tensor(self.img_raw_skeleton).float()
        if skeleton[0,1] >= 620:
            skeleton = torch.flip(skeleton, dims=[0])
        skeleton = torch.flip(skeleton, dims=[1])
        centerline = skeleton.detach().numpy()

        img = self.raw_img_rgb.copy()
        N = len(centerline)
        i = 0

        while i < N-10:
            pt1 = centerline[i]
            pt2 = centerline[i+10]

            point_color = (0, 255, 0)
            thickness = 1
            lineType = 4

            pt1_img = [int(pt1[0]), int(pt1[1])]
            pt2_img = [int(pt2[0]), int(pt2[1])] 

            cv2.circle(img, pt1_img, radius=1, color=(0,0,255), thickness=4)
            cv2.circle(img, pt2_img, radius=1, color=(0,0,255), thickness=4)
            cv2.line(img, pt1_img, pt2_img, point_color, thickness, lineType)

            i += 10

        plt.figure()
        plt.imshow(img)


    def plotverifycurvature(self):
        '''
        plot to see if the indexes correspondence is correct
        Based on OpenCV image writer

        Variables:
            self.raw_img_rgb ((640, 480) tensor): the image used as canvas for painting
            self.img_raw_skeleton: the skeleton points from the image
            self.p2d_gt_1000: the points find by bezier curve
            self.indexes: the indexes of the corresponding centerline points (p2d_gt_1000) about the skeleton
        '''
        img = self.raw_img_rgb.copy()

        skeleton = torch.as_tensor(self.img_raw_skeleton).float()
        if skeleton[0,1] >= 620:
            skeleton = torch.flip(skeleton, dims=[0])
        skeleton = torch.flip(skeleton, dims=[1])
        skeleton = skeleton.detach().numpy()

        centerline = self.p2d_gt_1000
        indexes = self.indexes
        N = len(indexes)

        for i in range(N):
            pt1 = skeleton[i]
            pt2 = centerline[indexes[i]]

            pt1_img = [int(pt1[0]), int(pt1[1])]
            pt2_img = [int(pt2[0]), int(pt2[1])]

            cv2.circle(img, pt1_img, radius=3, color=(0,0,255), thickness=4)
            cv2.circle(img, pt2_img, radius=3, color=(0,255,0), thickness=4)
            cv2.line(img, pt1_img, pt2_img, color=(255,0,0), thickness=1, lineType=4)

        filename = self.path + '/curve/' + str(self.GD_Iteration) + '.jpg'
        cv2.imwrite(filename, img)


    def Combine2Gif(self):
        '''
        Combine the jpgs in pictures folder into a mp4 video

        Variables:
            curve: The fitting process of curves
            eval: The learning process, loss evaluation
        '''
        writer_c = imageio.get_writer(self.path+'/videos/curve.mp4', fps=5)
        writer_e = imageio.get_writer(self.path+'/videos/eval.mp4', fps=5)

        for i in range(1,100):
            filename_c = os.path.join(self.path + '/curve/%d.jpg' %i)
            filename_e = os.path.join(self.path + '/eval/%d.jpg' %i)

            img_c =imageio.imread(filename_c)
            img_e =imageio.imread(filename_e)

            writer_c.append_data(img_c)
            writer_e.append_data(img_e)
        
        writer_c.close()
        writer_e.close()

    ##### ===========================================
    #       Learning Functions Group
    ##### =========================================== 

    def forwardProjection(self, control_pts):
        '''
        The forwardProjection of the learning steps

        Variables:
            self
        '''
        # get the points on the curve
        self.getBezier4pts(control_pts)
        # get the centerline projection
        self.proj_bezier_img = self.getProjPointCam(self.pos_bezier_cam, self.cam_K)
        # to check if there is any nan in the projected curve
        assert not torch.any(torch.isnan(self.proj_bezier_img))


    def getCostFun(self, ref_point_contour, P0_gt):

        P1 = P0_gt
        P2 = torch.zeros(3)
        C = torch.zeros(3)
        
        C[0], C[1], C[2] = self.para[0], self.para[1], self.para[2]
        P2[0], P2[1], P2[2] = self.para[3], self.para[4], self.para[5]

        P1_EM0inCam = P1 + self.OFF_SET
        P2_EM0inCam = P2 + self.OFF_SET
        self.C_EM0inCam = C + self.OFF_SET

        P1p = 2 / 3 * self.C_EM0inCam + 1 / 3 * P1_EM0inCam
        P2p = 2 / 3 * self.C_EM0inCam + 1 / 3 * P2_EM0inCam
        control_pts = torch.vstack((P1_EM0inCam, P1p, P2_EM0inCam, P2p))

        self.forwardProjection(control_pts)

        # The Obj obtained from the centerline
        obj_J_centerline, obj_J_tip = self.getCenterlineSegmentsObj()

        # The Obj obtained from the curvature
        obj_J_k = self.getCurvatureObj()

        # Chong's plotOptimizet function
        # self.drawlineandpts()

        obj_J = obj_J_centerline * self.loss_weight[0] + obj_J_tip * self.loss_weight[
               1] + obj_J_k * self.loss_weight[2]

        print("loss:", obj_J)

        if self.verbose:
            print("obj_J_all :", obj_J.detach().numpy())
        
        return obj_J

    def getOptimize(self, ref_point_contour, P0_gt):
        '''
        The main learning process

        Variables:
            self.optimizer: the torch optimizer
            self.loss: the value of loss calculated
            self.GD_Iteration: the number of iteration has been run
            self.verbose: to print the learning steps or not

        Returned Value:
            self.saved_opt_history
            self.para
        '''
        def closure():
            self.optimizer.zero_grad()
            self.loss = self.getCostFun(ref_point_contour, P0_gt)
            self.loss.backward()
            return self.loss
        
        self.optimizer = torch.optim.Adam([self.para], lr=1e-3)
        self.optimizer.zero_grad()
        loss_history = []
        last_loss = 99.0    # current loss value

        converge = False    # converge or not
        self.GD_Iteration = 0   # number of updates

        while not converge and self.GD_Iteration < self.total_itr:
            self.optimizer.step(closure)
            if (abs(self.loss - last_loss) < 1e-6):
                converge = True
            self.GD_Iteration += 1

            if self.verbose:
                print("Curr para : ", self.para)
                print("------------------------ FINISH ", self.GD_Iteration, " ^_^ STEP ------------------------ \n")
            
            last_loss = torch.clone(self.loss)
            loss_history.append(last_loss)

            saved_value = np.hstack((last_loss.detach().numpy(), self.para.detach().numpy()))
            self.saved_opt_history = np.vstack((self.saved_opt_history, saved_value))

            self.plotverifycurvature()

        self.error = torch.abs(self.para - self.para_gt)
        print("Final --->", self.para)
        print("GT    --->", self.para_gt)
        print("Error --->", self.error)

        return self.saved_opt_history, self.para



if __name__ == '__main__':
    ##### ===========================================
    #       Initialization
    ##### =========================================== 
    test_path = '/home/candice/Documents/ARCLab-CatheterControl/results/UN015/D00_0042'
    img_path = test_path + '/images/020.png'
    para_np = np.load(test_path+'/p3d_poses.npy')
    para = torch.from_numpy(para_np)
    para_init = torch.flatten(para[0])
    para_gt = torch.flatten(para[-1])
    para_init = torch.tensor(para_init, dtype=torch.float, requires_grad=True) 

    # loss weight
    loss_weight = torch.tensor([100.0, 0.0, 0.0])
    # total iteration for gradients optimization
    total_itr = 100

    # ground truth bezier length from P0 to P1
    curve_length_gt = 0.1906

    # ground truth bezier points: P0
    P0_gt = torch.tensor([0.02, 0.002, 0.0])

    ##### ===========================================
    #       Main reconstruction of bezier curve
    ##### ===========================================
    # Constructor of Class Reconstruction
    BzrCURVE = reconstructCurve(img_path, curve_length_gt, P0_gt, para_gt, para_init, loss_weight, total_itr)

    # # check if the order of centerline is right
    # # already checked
    # BzrCURVE.drawlineandpts()


    # do optimization
    BzrCURVE.getOptimize(None, P0_gt)

    # plot the final results
    BzrCURVE.plotProjCenterline()

    # combine the result to video
    BzrCURVE.Combine2Gif()

    # print final optimized parameters
    print('Optimized parameters', BzrCURVE.para)
        