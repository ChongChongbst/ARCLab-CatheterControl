import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.morphology import skeletonize

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import torch
import shutil
import os
import pdb
import argparse

np.set_printoptions(threshold=np.inf)

class reconstructCurve():
    def __init__(self, img_path, curve_length_gt, P0_gt, para_gt, para_init, loss_weight, total_itr, verbose=0):

        self.curve_length_gt = curve_length_gt
        self.P0_gt = P0_gt
        self.para_gt = para_gt
        self.para = para_init
        self.loss_weight = loss_weight
        self.total_itr = total_itr
        self.verbose = verbose

        self.OFF_SET = torch.tensor([0.00, 0.00, 0.00])

        downscale = 1.0
        # This doesn't make that big of a difference on synthetic images
        gaussian_blur_kern_size = 5
        dilate_iterations = 1

        # image size
        self.res_width = 640
        self.res_height = 480
        self.show_every_so_many_samples = 10
        self.R = 0.0013

        # camera E parameters
        cam_RT_H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        invert_y = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.cam_RT_H = torch.matmul(invert_y, cam_RT_H)

        # camera I parameters
        self.cam_K = torch.tensor([[883.00220751, 0.0, 320.0], [0.0, 883.00220751, 240.0], [0, 0, 1.0]])
        self.cam_K = self.cam_K / downscale
        self.cam_K[-1, -1] = 1

        self.Fourier_order_N = 1
        raw_img_rgb = cv2.imread(img_path)
        self.raw_img_rgb = cv2.resize(raw_img_rgb,
                                      (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
        self.raw_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

        self.optimizer = torch.optim.Adam([self.para], lr=1e-3)

        self.GD_Iteration = 0
        self.loss = None

        self.saved_opt_history = np.zeros((1, self.para.shape[0] + 1))

        ## get raw image skeleton upadate self.img_raw_skeleton
        self.getContourSamples()

        ## get ground truth 3D bezier curve
        self.pos_bezier_3D_gt = self.getAnyBezierCurve(self.para_gt, self.P0_gt)
        self.pos_bezier_3D_init = self.getAnyBezierCurve(para_init, self.P0_gt)


    def getOptimize(self, ref_point_contour, P0_gt):
        def closure():
            self.optimizer.zero_grad()
            self.loss = self.getCostFun(P0_gt)
            self.loss.backward()
            return self.loss

        self.optimizer.zero_grad()
        loss_history = []
        last_loss = 99.0  # current loss value

        converge = False  # converge or not
        self.GD_Iteration = 0  # number of updates

        while not converge and self.GD_Iteration < self.total_itr:

            self.optimizer.step(closure)

            if (abs(self.loss - last_loss) < 1e-6):
                converge = True

            self.GD_Iteration += 1

            if self.verbose:
                print("Curr para : ", self.para)
                print("------------------------ FINISH", self.GD_Iteration, "^_^ STEP ------------------------ \n")

            last_loss = torch.clone(self.loss)
            loss_history.append(last_loss)

            saved_value = np.hstack((last_loss.detach().numpy(), self.para.detach().numpy()))
            self.saved_opt_history = np.vstack((self.saved_opt_history, saved_value))

        print("Final --->", self.para)
        print("GT    --->", self.para_gt)
        print("Error --->", torch.abs(self.para - self.para_gt))

        return self.saved_opt_history, self.para

    def plotProjCenterline(self):
        centerline_draw_img_rgb = self.raw_img_rgb.copy()
        curve_3D_opt = self.pos_bezier_3D.detach().numpy()
        curve_3D_gt = self.pos_bezier_3D_gt.detach().numpy()
        curve_3D_init = self.pos_bezier_3D_init.detach().numpy()

        # chong test code
        centerline_from_img = torch.as_tensor(self.img_raw_skeleton).float()

        if centerline_from_img[0, 1] >= 620:
            centerline_from_img = torch.flip(centerline_from_img, dims=[0])

        centerline_from_img = torch.flip(centerline_from_img, dims=[1])

        # Draw centerline from skeleton on the image
        for i in range(centerline_from_img.shape[0] - 1):
            p1 = (int(centerline_from_img[i, 0]), int(centerline_from_img[i, 1]))
            p2 = (int(centerline_from_img[i + 1, 0]), int(centerline_from_img[i + 1, 1]))
            cv2.line(centerline_draw_img_rgb, p1, p2, (0, 255, 100), 4)

        # Draw bezier centerline
        for i in range(self.proj_bezier_img.shape[0] - 1):
            p1 = (int(self.proj_bezier_img[i, 0]), int(self.proj_bezier_img[i, 1]))
            p2 = (int(self.proj_bezier_img[i + 1, 0]), int(self.proj_bezier_img[i + 1, 1]))
            cv2.line(centerline_draw_img_rgb, p1, p2, (0, 100, 255), 4)

        # show
        # ----------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        ax = axes.ravel()
        ax[0].remove()
        ax[1].remove()
        ax[2].remove()
        ax[3].remove()
        # fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(cv2.cvtColor(centerline_draw_img_rgb, cv2.COLOR_BGR2RGB))
        ax0.set_title('Projected centerline')

        #the second subplot
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

        #the third subplot
        #training history
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(self.saved_opt_history[1:, 0], color='#6F69AC', linestyle='-', linewidth=1)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Loss')

        # plt.tight_layout()
        plt.show()

    def getContourSamples(self):
    	# update self.img_raw_skeleton to the centerline of the processed image
        ret, img_thresh = cv2.threshold(self.raw_img.copy(), 80, 255, cv2.THRESH_BINARY)
        self.img_thresh = img_thresh

        extend_dim = int(60)
        img_thresh_extend = np.zeros((self.res_height, self.res_width + extend_dim))
        img_thresh_extend[0:self.res_height, 0:self.res_width] = img_thresh.copy() / 255

        left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, self.res_width - 1]))
        left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, self.res_width - 10]))

        extend_vec_pt1_center = np.array([self.res_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1]) / 2])
        extend_vec_pt2_center = np.array(
            [self.res_width - 5, (left_boundarylineB_id[0] + left_boundarylineB_id[-1]) / 2])
        exten_vec = extend_vec_pt2_center - extend_vec_pt1_center
        
        if exten_vec[1]==0:
            exten_vec[1] += 0.00000001
        
        k_extend = exten_vec[0] / exten_vec[1]
        b_extend_up = self.res_width - k_extend * left_boundarylineA_id[0]
        b_extend_dw = self.res_width - k_extend * left_boundarylineA_id[-1]

        extend_ROI = np.array([
            np.array([self.res_width, left_boundarylineA_id[0]]),
            np.array([self.res_width, left_boundarylineA_id[-1]]),
            np.array([self.res_width + extend_dim, int(((self.res_width + extend_dim) - b_extend_dw) / k_extend)]),
            np.array([self.res_width + extend_dim,
                      int(((self.res_width + extend_dim) - b_extend_up) / k_extend)])
        ])
        
        img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

        # skletonize function return a binary image with skeleton 
        skeleton = skeletonize(img_thresh_extend)


        # By chong he
        # rearrange the skeleton points according to the order
        pts = np.argwhere(skeleton[:, 0:self.res_width] == 1)
        # get the indexes of abrupt indexes in the skeleton points
        abrupt_index = []
        i = 0
        while i < (len(pts)-1):
            if np.linalg.norm(pts[i]-pts[i+1], ord=np.inf, axis=None) > 2:
                checkpoint = pts[i]
                i += 1
                while np.linalg.norm(checkpoint-pts[i], ord=np.inf, axis=None) > 2 and i < (len(pts)-1):
                    abrupt_index.append(i)
                    i += 1
                #print(checkpoint)
            i += 1
        # store abrupt points in the list and delete the points from the skeleton points
        abrupt_pts = pts[abrupt_index]
        pts_temp = np.delete(pts,abrupt_index,axis=0)
        # arrange the abrupt points
        ind = np.argsort(abrupt_pts[:,1])
        abrupt_pts = abrupt_pts[ind]

        #print(pts_temp)
        skeleton_new = np.concatenate((abrupt_pts, pts_temp),axis=0)

        np.save("/home/candice/Documents/arclab_code/skeleton_reconstruct/skeleton.npy", skeleton_new)

        # insert the abrupt points to skeleton points according to the distances
        #print(pts)

        self.ref_skeleton = pts
        self.img_raw_skeleton = skeleton_new
        #By chonghe

        #self.img_raw_skeleton = np.argwhere(skeleton[:, 0:self.res_width] == 1)


        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
        # ax = axes.ravel()
        # ax[0].imshow(img_thresh_extend, cmap=plt.cm.gray)
        # ax[0].axis('off')
        # ax[0].set_title('original')
        # ax[1].imshow(skeleton, cmap=plt.cm.gray)
        # # ax[1].scatter(self.img_raw_skeleton[:, 1], self.img_raw_skeleton[:, 0], marker='o', s=1)
        # ax[1].plot(self.img_raw_skeleton[:, 1], self.img_raw_skeleton[:, 0], linestyle='-', linewidth=2)
        # ax[1].axis('off')
        # ax[1].set_title('skeleton')
        # # ax[1].plot(xxx[:, 1], xxx[:, 0], 'ro-', markersize=2, linewidth=1)
        # fig.tight_layout()
        # plt.show()
        # pdb.set_trace()

        ref_point1 = None

        return ref_point1

# to obtain an initial bezier curve
    def getAnyBezierCurve(self, para, P0):
        # get bezier curve through parameters

        num_samples = 200
        sample_list = torch.linspace(0, 1, num_samples)

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

# optimize the curve to fit the curve on the photo more
    def getCostFun(self, P0_gt):
        # GT values
        P1 = P0_gt
        P2 = torch.zeros(3)
        C = torch.zeros(3)
        # initialize the points
        C[0], C[1], C[2] = self.para[0], self.para[1], self.para[2]
        P2[0], P2[1], P2[2] = self.para[3], self.para[4], self.para[5]

        P1_EM0inCam = P1 + self.OFF_SET
        P2_EM0inCam = P2 + self.OFF_SET
        self.C_EM0inCam = C + self.OFF_SET

        P1p = 2 / 3 * self.C_EM0inCam + 1 / 3 * P1_EM0inCam
        P2p = 2 / 3 * self.C_EM0inCam + 1 / 3 * P2_EM0inCam
        control_pts = torch.vstack((P1_EM0inCam, P1p, P2_EM0inCam, P2p))

        self.forwardProjection(control_pts)

        obj_J_centerline, obj_J_tip, obj_J_bottom = self.getCenterlineSegmentsObj()

        obj_J_curveLength = self.getCurveLengthObj(self.curve_length_gt)

        obj_J = obj_J_centerline * self.loss_weight[0] + obj_J_tip * self.loss_weight[
            1] + obj_J_curveLength * self.loss_weight[2] 
            #+ obj_J_bottom * self.loss_weight[3]

        if self.verbose:
            print('obj_J_all :', obj_J.detach().numpy())

        return obj_J


    def forwardProjection(self, control_pts):
        # step 1 : get curve
        self.getBezierCurve(control_pts)
        # step 2 : get projection of center line
        self.proj_bezier_img = self.getProjPointCam(self.pos_bezier_cam, self.cam_K)
        assert not torch.any(torch.isnan(self.proj_bezier_img))


    def getDiscretizedCurvatureCurve(self, curvature_list, control_pts):
        '''
        # get the points calculated from discrete constant curvature with curvature list and control points
        # pos_discurve_3D: the 3D position points
        # the 2D position points in the pic
        # the 2D norm points
        '''
        self.num_samples = 200

        sample_list = torch.linspace(0, 1, self.num_samples)

        # Get positions and normals from samples along discrete constant curvature curve
        pos_discurve = torch.zeros(self.num_samples, 3)
        der_discurve = torch.zeros(self.num_samples, 3)

        for i, s in enumerate(sample_list):
            pos_discurve[i,:] = 



# four points in total to get bezier curve *Cubic Bezier Curves*
    def getBezierCurve(self, control_pts):
        '''
        # get the points calculated from bezier function with control_pts
        # pos_bezier_3D: the 3D position points 
        # pos_bezier_cam: the 2D position points in the pic 
        # der_bezier_cam: the 2D norm points 
        '''
        self.num_samples = 200
        P1 = control_pts[0, :]
        P1p = control_pts[1, :]
        P2 = control_pts[2, :]
        P2p = control_pts[3, :]

        sample_list = torch.linspace(0, 1, self.num_samples)

        # Get positions and normals from samples along bezier curve
        pos_bezier = torch.zeros(self.num_samples, 3)
        der_bezier = torch.zeros(self.num_samples, 3)

        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P1 + 3 * s * (1 - s)**2 * \
                P1p + 3 * (1 - s) * s**2 * P2p + s**3 * P2
            der_bezier[i, :] = -(1 - s)**2 * P1 + ((1 - s)**2 - 2 * s * (1 - s)) * P1p + (-s**2 + 2 *
                                                                                          (1 - s) * s) * P2p + s**2 * P2

        # Convert positions and normals to camera frame
        self.pos_bezier_3D = pos_bezier
        pos_bezier_H = torch.cat((pos_bezier, torch.ones(self.num_samples, 1)), dim=1)

        pos_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(pos_bezier_H, 0, 1)), 0, 1)
        self.pos_bezier_cam = pos_bezier_cam_H[1:, :-1]

        der_bezier_H = torch.cat((der_bezier, torch.zeros((self.num_samples, 1))), dim=1)
        der_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[1:, :], 0, 1)), 0,
                                           1)
        self.der_bezier_cam = der_bezier_cam_H[:, :-1]


    def getProjPointCam(self, p, cam_K):
        '''
        # project the points to camera frame
        '''
        # p is of size R^(Nx3)
        if p.shape == (3, ):
            p = torch.unsqueeze(p, dim=0)
        divide_z = torch.div(torch.transpose(p[:, :-1], 0, 1), p[:, -1])
        # print(p[:, -1].transpose(), '\n', '------------')
        divide_z = torch.cat((divide_z, torch.ones(1, p.shape[0])), dim=0)
        return torch.transpose(torch.matmul(cam_K, divide_z)[:-1, :], 0, 1)


    def getCenterlineSegmentsObj(self):
        '''
        # loss function
        # get the shape loss 
        # include the whole line and the tip point
        # ensure the centerline and img skeleton have the same shape
        '''
        centerline = torch.clone(self.proj_bezier_img)
        centerline = torch.flip(centerline, dims=[0])
        skeleton = torch.as_tensor(self.img_raw_skeleton).float()

        if skeleton[0, 1] >= 620:
            skeleton = torch.flip(skeleton, dims=[0])

        skeleton = torch.flip(skeleton, dims=[1])


        # Move by shifted starting
        # centerline_shift = centerline - (centerline[0, :] - skeleton[0, :])
        centerline_shift = centerline
        
        # Changed by Chong He
        # arrange the centerline points according to their distance with skeleton_ref
        centerline_by_corresp = []
        for i in range(skeleton.shape[0]):
            err = torch.linalg.norm(skeleton[i] - centerline_shift, ord=None, axis=1)
            index = torch.argmin(err) 
            temp = centerline_shift[index, ]
            centerline_by_corresp.append(temp)

        centerline_by_corresp = torch.stack(centerline_by_corresp)

        # code by chong he

        # test code written by chong he
        img1 = self.raw_img_rgb
        for i in range(centerline_by_corresp.shape[0] - 1):
            p1 = (int(centerline_by_corresp[i, 0]), int(centerline_by_corresp[i, 1]))
            p2 = (int(centerline_by_corresp[i + 1, 0]), int(centerline_by_corresp[i + 1, 1]))
            cv2.line(img1, p1, p2, (0, 255, 0), 4)
            cv2.circle(img1, p1, radius=1, color=(0,0,255), thickness=-1)

        for i in range(skeleton.shape[0] - 1):
            p1 = (int(skeleton[i, 0]), int(skeleton[i, 1]))
            p2 = (int(skeleton[i + 1, 0]), int(skeleton[i + 1, 1]))
            cv2.line(img1, p1, p2, (0, 255, 0), 4)
            cv2.circle(img1, p1, radius=1, color=(0,0,255), thickness=-1)
        plt.figure(1)
        plt.imshow(img1)
        plt.show()
        # test code written by chong he

        self.CENTERLINE_SHAPE = centerline.shape[0]
        # err_skeleton_by_corresp = torch.linalg.norm(skeleton - skeleton_by_corresp, ord=None, axis=1) / self.res_width
        err_centerline_by_corresp = torch.linalg.norm(skeleton - centerline_by_corresp, ord=None, axis=1) / 1.0
        err_centerline_sum_by_corresp = torch.sum(err_centerline_by_corresp) / self.CENTERLINE_SHAPE
        # -------------------------------------------------------------------------------

        err_obj_Tip = torch.linalg.norm(skeleton[0, :] - centerline[0, :], ord=None)

        # loss by chong he
        err_obj_Bottom = torch.linalg.norm(skeleton[-1, :] - centerline[-1, :], ord=None)

        return err_centerline_sum_by_corresp, err_obj_Tip, err_obj_Bottom

    def getCurveLengthObj(self, curve_length_gt):
        curve_3d = self.pos_bezier_cam
        diff_curve = torch.diff(curve_3d, axis=0)
        len_diff = torch.linalg.norm(diff_curve, ord=None, axis=1)
        len_sum = torch.sum(len_diff, dim=0)

        obj_J_curveLen = torch.abs(len_sum - curve_length_gt) * (1.0 / curve_length_gt)

        # pdb.set_trace()

        return obj_J_curveLen

    def CollectNeighboringPoints(pc, dc, P, dM, r0):
        P0 = []
        d0 = []
        L0 = []
        return P0, d0, L0
    
    def OrderingPoints(dir, p0, P):
        # 
        d_initial = 1
        d_increment = 1
        d_max_break = 200

        pc = p0
        dc = d_initial
        ordered_P = []
        pl = []
        while dc <= d_max_break:
            P_a, P_b = np.ndarray([]), np.ndarray([])
            pts_0, r, L0 = reconstructCurve.CollectNeighboringPoints(pc, dc, P, dM, r0)
            for pi in pts_0:
                # get direction vector of Line L0
                dir_L = np.inner()
                # get direction vector of line pc-pi
                dir_P = pi - pc
                if np.inner(dir_L, dir_P) >= 0:
                    P_a = np.append(P_a, pi)
                else:
                    P_b = np.append(P_b, pi)

        return 0


if __name__ == '__main__':
    ##### ===========================================
    #       Initialization
    ##### ===========================================
    ###image path
    #img_path = "/home/fei/ARCLab-CCCatheter/data/rendered_images/dof2_64/dof2_c40_0.0005_-0.005_0.2_0.01.png"

    ###image path
    ###img_path = "/home/fei/ARCLab-CCCatheter/data/rendered_images/dof2_64/dof2_c40_0.0005_-0.005_0.2_0.01.png"


    ##### ===========================================
    ###image path
    #img_path = "/home/fei/ARCLab-CCCatheter/data/rendered_images/dof2_64/dof2_c40_0.0005_-0.005_0.2_0.01.png"
    #### 64/dof2_c40_0.0005_-0.005_0.2_0.01.png"
    #### 64/dof2_c40_0.0005_-0.005_0.2_0.01.png"


    img_path = '/home/candice/Documents/Pics/recon_test_data/0000_seg.png'

    ### ground truth bezier curve length from P0 to P1
    curve_length_gt = 0.1906

    ### ground truth bezier points : P0
    P0_gt = torch.tensor([2e-2, 2e-3, 0.0])

    ### ground truth bezier points : [PC, P1]
    para_gt = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.19168896], dtype=torch.float)

    ### initialized bezier points : [P0, PC, P1]
    para_init = torch.tensor([0.01957763, 0.00191553, 0.09690971, -0.03142124, -0.00828425, 0.18168159],
                             dtype=torch.float,
                             requires_grad=True)
    # para_init = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.19168896],
    #                          dtype=torch.float,
    #                          requires_grad=True)

    ### initialized weights for three different loss
    loss_weight = torch.tensor([1.0, 100.0, 0.0, 1.0])

    ### total itr for gradients optimization
    total_itr = 100

    ##### ===========================================
    #       Main reconstruction of bezier curve
    ##### ===========================================
    ### constructor of Class reconstructCurve
    BzrCURVE = reconstructCurve(img_path, curve_length_gt, P0_gt, para_gt, para_init, loss_weight, total_itr)

    ### do optimization
    BzrCURVE.getOptimize(None, P0_gt)

    # test
    # print(BzrCURVE.proj_bezier_img.shape)
    # print(BzrCURVE.img_raw_skeleton.shape)

    ### plot the final results
    BzrCURVE.plotProjCenterline()

    ###  print final optimized parameters
    print('Optimized parameters', BzrCURVE.para)
