import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = [1280, 800]
# mpl.rcParams['figure.dpi'] = 300

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



class reconstructCurve():
    def __init__(self, img_path, curve_length_gt, para_gt, para_init, loss_weight, total_itr):

        # self.img_id = 1
        # self.save_dir = './steps_imgs_left_1_STCF'
        # if os.path.isdir(self.save_dir):
        #     shutil.rmtree(self.save_dir)
        # os.mkdir(self.save_dir)
        # os.mkdir(self.save_dir + '/centerline')
        # os.mkdir(self.save_dir + '/contours')

        self.curve_length_gt = curve_length_gt
        self.para_gt = para_gt
        self.para_init = para_init
        self.para = para_init
        self.loss_weight = loss_weight
        self.total_itr = total_itr
        self.OFF_SET = torch.tensor([0.00, 0.00, 0.00])

        # self.img_raw_skeleton = np.genfromtxt(
        #     "/home/fei/catheter_reconstruction_ws/saved_images_calibration_case1/seg_images_calibration_case1/seg_left_recif_1_skeleton.csv",
        #     delimiter=',')

        # img_path = "../exp_data_dvrk/seg_video5/seg_left_recif_0.png"
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
        # self.cam_distCoeffs = torch.tensor([-4.0444238705587998e-01, 5.8161897902897197e-01, -4.9797819387316098e-03, 2.3217574337593299e-03, -2.1547479006608700e-01])
        # raw_img_rgb_undst = cv2.undistort(raw_img_rgb, self.cam_K.detach().numpy(), self.cam_distCoeffs.detach().numpy())
        self.raw_img_rgb = cv2.resize(raw_img_rgb,
                                      (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
        self.raw_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

        # self.blur_raw_img = cv2.GaussianBlur(self.raw_img, (gaussian_blur_kern_size, gaussian_blur_kern_size), 0)
        # edges_img = canny(self.blur_raw_img, 2, 1, 100)
        # self.edges_img = cv2.dilate(edges_img.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=dilate_iterations)

        # self.optimizer = torch.optim.SGD([self.para], lr=1e-5)
        self.optimizer = torch.optim.Adam([self.para], lr=1e-3)

        # self.optimal_R_EMraw2Cam = torch.tensor([[0.40533652, -0.91020415, 0.08503356],
        #                                          [0.86140179, 0.41142924, 0.29784715],
        #                                          [-0.30608701, -0.04748027, 0.95081879]])
        # self.optimal_t_EMraw2Cam = torch.tensor([[-0.120146], [-0.20414568], [0.22804266]])
        self.GD_Iteration = 0
        self.loss = None

        self.saved_opt_history = np.zeros((1, self.para.shape[0] + 1))

        ## get raw image skeleton
        self.getContourSamples()

        ## get ground truth 3D bezier curve
        self.pos_bezier_3D_gt = self.getAnyBezierCurve(self.para_gt)
        self.pos_bezier_3D_init = self.getAnyBezierCurve(para_init)

    def getBezierCurve(self, control_pts):

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

        # print(pos_bezier)
        # pos_bezier.register_hook(print)
        # P1.register_hook(print)
        # P1p.register_hook(print)
        # P2.register_hook(print)
        # P2p.register_hook(print)

        der_bezier_H = torch.cat((der_bezier, torch.zeros((self.num_samples, 1))), dim=1)
        der_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[1:, :], 0, 1)), 0,
                                           1)
        self.der_bezier_cam = der_bezier_cam_H[:, :-1]

    def getAnyBezierCurve(self, para):

        num_samples = 200
        sample_list = torch.linspace(0, 1, num_samples)

        P1 = torch.tensor([para[0], para[1], para[2]])
        PC = torch.tensor([para[3], para[4], para[5]])
        P2 = torch.tensor([para[6], para[7], para[8]])
        P1p = 2 / 3 * PC + 1 / 3 * P1
        P2p = 2 / 3 * PC + 1 / 3 * P2

        # Get positions and normals from samples along bezier curve
        pos_bezier = torch.zeros(num_samples, 3)
        der_bezier = torch.zeros(num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P1 + 3 * s * (1 - s)**2 * \
                P1p + 3 * (1 - s) * s**2 * P2p + s**3 * P2

        return pos_bezier

    def getProjPointCam(self, p, cam_K):
        # p is of size R^(Nx3)
        if p.shape == (3, ):
            p = torch.unsqueeze(p, dim=0)

        divide_z = torch.div(torch.transpose(p[:, :-1], 0, 1), p[:, -1])

        # print(p[:, -1].transpose(), '\n', '------------')

        divide_z = torch.cat((divide_z, torch.ones(1, p.shape[0])), dim=0)

        # print(torch.matmul(cam_K, divide_z)[:-1, :])
        # pdb.set_trace()
        return torch.transpose(torch.matmul(cam_K, divide_z)[:-1, :], 0, 1)

    def isPointInImage(self, p_proj, width, height):
        if torch.all(torch.isnan(p_proj)):
            # print('NaN')
            return False
        if p_proj[0] < 0 or p_proj[1] < 0 or p_proj[0] > width or p_proj[1] > height:
            # print('Out')
            return False

        return True

    # position and direction must be 3D and defined in the camera frame.
    def getProjCylinderLines(self, position, direction, R, fx, fy, cx, cy):
        a = direction[0]
        b = direction[1]
        c = direction[2]

        x0 = position[0]
        y0 = position[1]
        z0 = position[2]

        alpha1 = (1 - a * a) * x0 - a * b * y0 - a * c * z0
        beta1 = -a * b * x0 + (1 - b * b) * y0 - b * c * z0
        gamma1 = -a * c * x0 - b * c * y0 + (1 - c * c) * z0

        alpha2 = c * y0 - b * z0
        beta2 = a * z0 - c * x0
        gamma2 = b * x0 - a * y0

        C = x0 * x0 + y0 * y0 + z0 * z0 - \
            (a * x0 + b * y0 + c * z0) * (a * x0 + b * y0 + c * z0) - R * R

        if C < 0:
            print("Recieved C less than 0")
            e_1 = torch.tensor([-999.0, -999.0])
            e_2 = torch.tensor([-999.0, -999.0])
            # line1 = torch.stack((float('nan'), float('nan'), float('nan')), dim=0)
            # line2 = torch.stack((float('nan'), float('nan'), float('nan')), dim=0)
            line1 = torch.tensor([-999.0, -999.0, -999.0])
            line2 = torch.tensor([-999.0, -999.0, -999.0])
            # return (-1, -1), (-1, -1), (float('nan'), float('nan'), float('nan')), (float('nan'), float('nan'), float('nan'))
            return e_1, e_2, line1, line2

        temp = R / math.sqrt(C)

        k1 = (alpha1 * temp - alpha2)
        k2 = (beta1 * temp - beta2)
        k3 = (gamma1 * temp - gamma2)

        # Get edges! Fu + Gv = D convert to hough trnasform cos(theta)u + sin(theta)v = r
        F = k1 / fx
        G = k2 / fy
        D = -k3 + F * cx + G * cy

        # Distance can be negative in our HT world,(0.0, 0.0, 255.0)
        if D < 0:
            D = -D
            G = -G
            F = -F

        # store for intersection point
        # line1 = (F, G, -D)
        line1 = torch.stack((F, G, -D), dim=0)

        # arctan2 range is -pi, pi so maybe switch to arctan which is -pi/2.0 to pi/2.0
        # e_1 = (D / math.sqrt(F * F + G * G), np.arctan2(G, F))
        # e_1 = (-D / torch.sqrt(F * F + G * G), torch.atan(G / F))

        e_1 = torch.stack((-D / torch.sqrt(F * F + G * G), torch.atan(G / F)), dim=0)

        k1 += 2 * alpha2
        k2 += 2 * beta2
        k3 += 2 * gamma2

        F = k1 / fx
        G = k2 / fy
        D = -k3 + F * cx + G * cy

        if D < 0:
            D = -D
            G = -G
            F = -F

        # store for intersection point
        # line2 = (F, G, -D)
        line2 = torch.stack((F, G, -D), dim=0)
        # print(line2)

        # e_2 = (D / math.sqrt(F * F + G * G), np.arctan2(G, F))
        # e_2 = (-D / torch.sqrt(F * F + G * G), torch.atan(G / F))
        e_2 = torch.stack((-D / torch.sqrt(F * F + G * G), torch.atan(G / F)), dim=0)

        return e_1, e_2, line1, line2

    def getProjCylinderImage(self, proj_bezier_img):

        list_of_edges_1 = []
        list_of_edges_2 = []
        list_of_edges1_ABC = []
        list_of_edges2_ABC = []

        # Draw projected cylinder
        cylinder_draw_img_rgb = self.raw_img_rgb.copy()
        # cylinder_draw_img_rgb = []

        for i, p in enumerate(proj_bezier_img):

            # If the centerline point is out of image frame
            # if not self.isPointInImage(p, cylinder_draw_img_rgb.shape[1], cylinder_draw_img_rgb.shape[0]):
            #     # print("[WARN] bezier center out of image! --> ", i)
            #     self.first_outofimage_bezier = i
            #     continue

            # if (i % self.show_every_so_many_samples != 0) and (i != proj_bezier_img.shape[0] - 1) and (i != self.first_outofimage_bezier + 1):
            if (i % self.show_every_so_many_samples != 0) and (i != proj_bezier_img.shape[0] - 1):
                continue

            # print(i)

            # s_i = float(i) / float(self.num_samples - 1) * 1

            centpnt_on_cylinder = self.pos_bezier_cam[i, :]
            tangdir_on_cylinder = self.der_bezier_cam[i] / \
                torch.linalg.norm(self.der_bezier_cam[i])

            e_1, e_2, line1, line2 = self.getProjCylinderLines(centpnt_on_cylinder, tangdir_on_cylinder, self.R,
                                                               self.cam_K[0, 0], self.cam_K[1, 1], self.cam_K[0, -1],
                                                               self.cam_K[1, -1])

            # if e_1 == (-1, -1) and e_2 == (-1, -1):
            #     continue

            # a = np.cos(e_1[1])
            # b = np.sin(e_1[1])
            # x0 = a * e_1[0]
            # y0 = b * e_1[0]
            # x1 = int(x0 + 2000 * (-b))
            # y1 = int(y0 + 2000 * (a))
            # x2 = int(x0 - 2000 * (-b))
            # y2 = int(y0 - 2000 * (a))

            # cv2.line(cylinder_draw_img_rgb, (x1, y1), (x2, y2), (230, 0, 0), 2)

            # a = np.cos(e_2[1])
            # b = np.sin(e_2[1])
            # x0 = a * e_2[0]
            # y0 = b * e_2[0]
            # x1 = int(x0 + 2000 * (-b))
            # y1 = int(y0 + 2000 * (a))
            # x2 = int(x0 - 2000 * (-b))
            # y2 = int(y0 - 2000 * (a))

            # cv2.line(cylinder_draw_img_rgb, (x1, y1), (x2, y2), (0, 0, 230), 2)

            # if e_1 == (-1, -1) and e_2 == (-1, -1):
            #     continue
            # x1 = int(0)
            # y1 = int(-line1[2] / line1[1])
            # x2 = int(self.res_width)
            # y2 = int(-(line1[0] * x2 + line1[2]) / line1[1])
            # cv2.line(cylinder_draw_img_rgb, (x1, y1), (x2, y2), (230, 0, 230), 1)
            # x1 = int(0)
            # y1 = int(-line2[2] / line2[1])
            # x2 = int(self.res_width)
            # y2 = int(-(line2[0] * x2 + line2[2]) / line2[1])
            # cv2.line(cylinder_draw_img_rgb, (x1, y1), (x2, y2), (0, 230, 230), 1)

            list_of_edges_1.append(e_1)
            list_of_edges_2.append(e_2)
            list_of_edges1_ABC.append(line1)
            list_of_edges2_ABC.append(line2)

        self.list_of_edges_1 = torch.stack(list_of_edges_1)
        self.list_of_edges_2 = torch.stack(list_of_edges_2)
        self.list_of_edges1_ABC = torch.stack(list_of_edges1_ABC)
        self.list_of_edges2_ABC = torch.stack(list_of_edges2_ABC)

        # print(self.list_of_edges_2)
        # self.AAA = self.list_of_edges2_ABC[-1, 0]
        # self.AAA = list_of_edges2_ABC[-1][0]
        # self.AAA = centpnt_on_circle
        # print(self.AAA, '+++++-----------------------------------')

        # self.list_of_edges2_ABC.register_hook(print)
        assert not torch.any(torch.isnan(self.list_of_edges1_ABC))
        assert not torch.any(torch.isnan(self.list_of_edges2_ABC))

        return cylinder_draw_img_rgb

    def getProjCirclesImage(self, proj_bezier_img):

        # Draw projected cylinder
        circles_draw_img_rgb = self.raw_img_rgb.copy()
        # circles_draw_img_rgb = []

        list_of_ellipses = []
        list_of_ellipses_K = []

        for i, p in enumerate(proj_bezier_img):
            # If the centerline point is out of image frame
            # if not self.isPointInImage(p, circles_draw_img_rgb.shape[1], circles_draw_img_rgb.shape[0]):
            #     continue

            # if (i % self.show_every_so_many_samples != 0) and (i != proj_bezier_img.shape[0] - 1) and (i != self.first_outofimage_bezier + 1):
            if (i % self.show_every_so_many_samples != 0) and (i != proj_bezier_img.shape[0] - 1):
                continue

            # s_i = float(i) / float(self.num_samples - 1) * 1

            centpnt_on_circle = self.pos_bezier_cam[i, :]
            # tangdir_on_circle = self.der_bezier_cam[i] / np.linalg.norm(self.der_bezier_cam[i])
            tangdir_on_circle = self.der_bezier_cam[i]

            # print(centpnt_on_circle, i)
            # print(tangdir_on_circle)

            # cx, cy, amaj, amin, angle, k0_i, k1_i, k2_i, k3_i, k4_i, k5_i = self.getProjCircles(centpnt_on_circle, tangdir_on_circle, self.R, self.cam_K[0, 0], self.cam_K[1, 1], self.cam_K[0, -1],
            #                                                                                     self.cam_K[1, -1])
            ellipese_v, ellipese_k = self.getProjCircles(centpnt_on_circle, tangdir_on_circle, self.R, self.cam_K[0, 0],
                                                         self.cam_K[1, 1], self.cam_K[0, -1], self.cam_K[1, -1], i)
            # print(i)
            # print(cx, cy, amaj, amin, angle, k0_i, k1_i, k2_i, k3_i, k4_i, k5_i, '\n')

            # if torch.any(torch.isnan(ellipese_v)):
            if torch.any(torch.isnan(ellipese_v)):
                # print('[WARN] This bezier point get NaN circle! ---->', i)
                continue
            if torch.any(torch.isinf(ellipese_v)):
                # print('[WARN] This bezier point get NaN circle! ---->', i)
                continue

            # if i < 10:
            # self.AAA = ellipese_k
            # print(i, self.AAA)

            # if torch.isnan(cx) or torch.isnan(cy) or torch.isnan(amaj) or torch.isnan(amin):
            #     print('[WARN] This bezier point get NaN circle! ---->', i)
            #     continue

            # list_of_ellipses.append(torch.stack((cx, cy, amaj, amin, angle), dim=0))
            # list_of_ellipses_K.append(torch.stack((k0_i, k1_i, k2_i, k3_i, k4_i, k5_i), dim=0))
            list_of_ellipses.append(ellipese_v)
            list_of_ellipses_K.append(ellipese_k)

            # print(ellipese_k)

        self.list_of_ellipses = torch.stack(list_of_ellipses)
        self.list_of_ellipses_K = torch.stack(list_of_ellipses_K)

        assert not torch.any(torch.isnan(self.list_of_ellipses_K))

        # self.AAA = list_of_ellipses_K
        # # # self.AAA = centpnt_on_circle
        # print(self.AAA, '+++++-----------------------------------')

        return circles_draw_img_rgb

    # position and direction must be 3D and defined in the camera frame.
    def getProjCircles(self, position, direction, R, fx, fy, cx, cy, i):
        alpha = direction[0]
        beta = direction[1]
        gamma = direction[2]

        x0 = position[0]
        y0 = position[1]
        z0 = position[2]

        abc = alpha * x0 + beta * y0 + gamma * z0
        # if abc == 0:
        #     abc += 1e-16

        a = alpha / abc
        b = beta / abc
        c = gamma / abc

        # print(alpha, x0, beta, y0, gamma, z0, '++++++++++++')

        # k in camera frame
        k0_c = (a * a) * (x0 * x0 + y0 * y0 + z0 * z0 - R * R) + 1 - 2 * a * x0
        k1_c = (b * b) * (x0 * x0 + y0 * y0 + z0 * z0 - R * R) + 1 - 2 * b * y0
        k2_c = (a * b) * (x0 * x0 + y0 * y0 + z0 * z0 - R * R) - b * x0 - a * y0
        k3_c = (a * c) * (x0 * x0 + y0 * y0 + z0 * z0 - R * R) - c * x0 - a * z0
        k4_c = (b * c) * (x0 * x0 + y0 * y0 + z0 * z0 - R * R) - c * y0 - b * z0
        k5_c = (c * c) * (x0 * x0 + y0 * y0 + z0 * z0 - R * R) + 1 - 2 * c * z0

        # k in image frame
        k0_i = k0_c / (fx * fx)
        k1_i = k1_c / (fy * fy)
        k2_i = k2_c / (fx * fy)
        k3_i = k3_c / fx - k0_c * cx / (fx * fx) - k2_c * cy / (fx * fy)
        k4_i = k4_c / fy - k1_c * cy / (fy * fy) - k2_c * cx / (fx * fy)
        k5_i = k5_c + k0_c * cx * cx / (fx * fx) + k1_c * cy * cy / (fy * fy) + 2 * k2_c * cx * cy / (
            fx * fy) - 2 * k3_c * cx / fx - 2 * k4_c * cy / fy

        # xc = (k1_i * k3_i - k2_i * k4_i) / (k2_i * k2_i - k0_i * k1_i)
        # yc = (k0_i * k4_i - k2_i * k3_i) / (k2_i * k2_i - k0_i * k1_i)

        # e = (k1 - k0 + np.sqrt((k1 - k0)**2 + 4 * k2 * k2)) / (2 * k2)
        # asq = 2 * (k0 * xc * xc + 2 * k2 * xc * yc + k1 * yc * yc - k5) / (k0 + k1 + np.sqrt((k1 - k0)**2 + 4 * k2 * k2))
        # bsq = 2 * (k0 * xc * xc + 2 * k2 * xc * yc + k1 * yc * yc - k5) / (k0 + k1 - np.sqrt((k1 - k0)**2 + 4 * k2 * k2))

        # https://en.wikipedia.org/wiki/Ellipse
        A = k0_i
        B = 2 * k2_i
        C = k1_i
        D = 2 * k3_i
        E = 2 * k4_i
        F = k5_i

        delta = (A * C - B**2) * F + (B * E * D) / \
            4 - (C * D * D) / 4 - (A * E * E) / 4

        # print(A, B, C, D, E, F)

        # if B4AC == 0:
        #     print("Recieved B * B - 4 * A * C less than 0")
        #     # return torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(
        #     #     float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))
        #     B4AC += 1e-4

        B4AC = B * B - 4 * A * C
        # assert not torch.allclose(B4AC, torch.zeros_like(B4AC)), f"Dividing: {B4AC}"

        # print(C * delta, B4AC)

        # if B4AC >= 0:
        #     B4AC = torch.maximum(B4AC, B4AC * 0 + 1e-6)
        # else:
        #     B4AC = torch.minimum(B4AC, B4AC * 0 - 1e-6)

        x_ellipse = (2 * C * D - B * E) / B4AC
        y_ellipse = (2 * A * E - B * D) / B4AC

        # major_axis = (-torch.sqrt(2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (A + C + torch.sqrt((A - C)**2 + B * B)))) / B4AC
        # minor_axis = (-torch.sqrt(2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (A + C - torch.sqrt((A - C)**2 + B * B)))) / B4AC

        tmp1 = (A - C)**2 + B * B
        # if tmp1 == 0:
        #     tmp1 = 1e-16

        tmp2 = A + C - torch.sqrt(tmp1)
        # if tmp2 <= 0:
        #     tmp2 = 1e-16

        tmp3 = 2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (A + C + torch.sqrt(tmp1))
        tmp4 = 2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (tmp2)
        if tmp3 == 0:
            # print("0 tmps3")
            tmp3 += 1e-16
        if tmp4 == 0:
            tmp4 += 1e-16

        # print("tmps : ", A * E * E + C * D * D - B * D * E + B4AC * F, tmp3, tmp4)
        major_axis = (-torch.sqrt(tmp3)) / B4AC
        minor_axis = (-torch.sqrt(tmp4)) / B4AC

        # print(2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (A + C + torch.sqrt((A - C)**2 + B * B)), '++++')

        # major_axis = (-np.sqrt(2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (A + C + np.sqrt((A - C)**2 + B * B)))) / B4AC
        # minor_axis = (-np.sqrt(2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (A + C - np.sqrt((A - C)**2 + B * B)))) / B4AC

        if B != 0:
            # angle = np.arctan2(C - A - torch.sqrt((A - C)**2 + B * B), B)
            angle = torch.atan((C - A - torch.sqrt((A - C)**2 + B * B)) / B)
        if B == 0 and A <= C:
            angle = torch.tensor(0.0)
        if B == 0 and A > C:
            angle = torch.tensor(np.pi / 2)

        ellipse_V = torch.stack((x_ellipse, y_ellipse, major_axis, minor_axis, angle), dim=0)
        ellipse_K = torch.stack((k0_i, k1_i, k2_i, k3_i, k4_i, k5_i), dim=0)

        # print(x_ellipse, y_ellipse, major_axis, minor_axis, angle)
        # print(ellipse)
        # print(ellipse_K)

        # if math.isnan(major_axis):
        #     print('tmp3/B * B - 4 * A * C', tmp3, B * B - 4 * A * C)

        # if i < 20:
        #     #     self.AAA = minor_axis
        #     #     print('+++++++', 2 * (A * E * E + C * D * D - B * D * E + B4AC * F) * (A + C - torch.sqrt((A - C)**2 + B * B + 1e-16)))
        #     #     print(i, self.AAA)
        #     print('----->', A + C - torch.sqrt((A - C)**2 + B * B))
        #     print('----->---', (A - C)**2 + B * B)

        # return x_ellipse, y_ellipse, major_axis, minor_axis, angle, k0_i, k1_i, k2_i, k3_i, k4_i, k5_i
        return ellipse_V, ellipse_K

    def getProjIntersectionCircleCylinder(self):

        list_of_inters_point1 = []
        list_of_inters_point2 = []

        # print(self.list_of_ellipses_K)
        # print(self.list_of_edges1_ABC)
        # print(self.list_of_edges2_ABC)
        # print(self.list_of_ellipses)
        # print(self.list_of_ellipses_K.shape[0])

        last_notOnImage_point1 = torch.FloatTensor()
        last_notOnImage_point2 = torch.FloatTensor()

        for i in range(self.list_of_ellipses_K.shape[0]):
            # if torch.any(torch.isnan(self.list_of_ellipses_K[i, :])):
            #     # print('[WARN] This bezier point get NaN circle! ---->', i)
            #     continue
            # if torch.all(torch.isnan(self.list_of_edges1_ABC[i, :])):
            #     print('[WARN] This bezier point get NaN circle! ---->', i)
            #     continue
            # if torch.all(torch.isnan(self.list_of_edges2_ABC[i, :])):
            #     print('[WARN] This bezier point get NaN circle! ---->', i)
            # #     continue
            # print(torch.all(torch.eq(self.list_of_edges1_ABC[i, :], torch.tensor([-999.0, -999.0, -999.0]))))
            # print(self.list_of_edges1_ABC[i, :])

            # print(self.list_of_edges1_ABC)

            if torch.all(torch.eq(self.list_of_edges1_ABC[i, :], torch.tensor([-999.0, -999.0, -999.0]))):
                continue
            if torch.all(torch.eq(self.list_of_edges2_ABC[i, :], torch.tensor([-999.0, -999.0, -999.0]))):
                continue

            A1 = self.list_of_edges1_ABC[i, 0]
            B1 = self.list_of_edges1_ABC[i, 1]
            C1 = self.list_of_edges1_ABC[i, 2]

            A2 = self.list_of_edges2_ABC[i, 0]
            B2 = self.list_of_edges2_ABC[i, 1]
            C2 = self.list_of_edges2_ABC[i, 2]

            K0 = self.list_of_ellipses_K[i, 0]
            K1 = self.list_of_ellipses_K[i, 1]
            K2 = self.list_of_ellipses_K[i, 2]
            K3 = self.list_of_ellipses_K[i, 3]
            K4 = self.list_of_ellipses_K[i, 4]
            K5 = self.list_of_ellipses_K[i, 5]

            alpha1 = B1**2 * K0 + A1**2 * K1 - 2 * A1 * B1 * K2
            beta1 = 2 * A1 * C1 * K1 - 2 * K2 * B1 * C1 + 2 * K3 * B1**2 - 2 * K4 * A1 * B1
            gamma1 = K1 * C1**2 - 2 * K4 * B1 * C1 + K5 * B1**2
            x1 = -beta1 / (2 * alpha1)
            y1 = -(A1 * x1 + C1) / B1
            # print(x1, y1)

            alpha2 = B2**2 * K0 + A2**2 * K1 - 2 * A2 * B2 * K2
            beta2 = 2 * A2 * C2 * K1 - 2 * K2 * B2 * C2 + 2 * K3 * B2**2 - 2 * K4 * A2 * B2
            gamma2 = K1 * C2**2 - 2 * K4 * B2 * C2 + K5 * B2**2

            # print(beta2**2 - 4 * alpha2 * gamma2)
            x2 = -beta2 / (2 * alpha2)
            y2 = -(A2 * x2 + C2) / B2

            # print(x1, y1, i, '+++++')

            candi_xy1 = torch.stack([x1, y1], dim=0)
            candi_xy2 = torch.stack([x2, y2], dim=0)

            if self.isPointInImage(candi_xy1, self.res_width, self.res_height):
                list_of_inters_point1.append(candi_xy1)
            else:
                last_notOnImage_point1 = candi_xy1
                # print('out1')

            if self.isPointInImage(candi_xy2, self.res_width, self.res_height):
                list_of_inters_point2.append(candi_xy2)
            else:
                last_notOnImage_point2 = candi_xy2
                # print('out2')

        if last_notOnImage_point1.shape[0] == 0:
            last_notOnImage_point1 = list_of_inters_point1[0]
        if last_notOnImage_point2.shape[0] == 0:
            last_notOnImage_point2 = list_of_inters_point2[0]
            # print(last_notOnImage_point1)

        # print(list_of_inters_point1)
        # print(list_of_inters_point2)

        self.last_notOnImage_point1 = last_notOnImage_point1
        self.last_notOnImage_point2 = last_notOnImage_point2
        self.list_of_inters_point1 = torch.stack(list_of_inters_point1)
        self.list_of_inters_point2 = torch.stack(list_of_inters_point2)

        # print(self.last_notOnImage_point1)

        # print(self.list_of_edges1_ABC)
        # print(self.list_of_inters_point1[0, :])
        # print(self.last_notOnImage_point2)
        # print(self.list_of_inters_point1)
        # print(self.list_of_inters_point2)
        # print('++++++++++++++++++++++++++++++++++++\n')

        # print(self.list_of_inters_point2.shape)
        # print(bbbbbb)
        # self.AAA = abs(self.list_of_inters_point2[-1, 0])
        # print('++++++++++++++++++++++++++++++++++++\n')

        assert not torch.any(torch.isnan(self.list_of_inters_point1))
        assert not torch.any(torch.isnan(self.list_of_inters_point2))

        # self.list_of_inters_point2.register_hook(print)

    def getContourSamples(self):

        # img_contour = self.edges_img.copy()

        ret, img_thresh = cv2.threshold(self.raw_img.copy(), 80, 255, cv2.THRESH_BINARY)

        # self.img_thresh = cv2.bitwise_not(img_thresh)
        self.img_thresh = img_thresh

        # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        # ax = axes.ravel()
        # ax[0].imshow(self.raw_img, cmap=cm.gray)
        # ax[0].set_title('Input image')
        # ax[1].imshow(self.img_thresh, cmap=cm.gray)
        # ax[1].set_title('img_thresh image')
        # plt.show()

        # img_thresh = cv2.adaptiveThreshold(self.raw_img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # print(self.raw_img)

        # perform skeletonization, need to extend the boundary of the image
        extend_dim = int(60)
        img_thresh_extend = np.zeros((self.res_height, self.res_width + extend_dim))
        img_thresh_extend[0:self.res_height, 0:self.res_width] = img_thresh.copy() / 255

        left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, self.res_width - 1]))
        left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, self.res_width - 10]))

        extend_vec_pt1_center = np.array([self.res_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1]) / 2])
        extend_vec_pt2_center = np.array(
            [self.res_width - 5, (left_boundarylineB_id[0] + left_boundarylineB_id[-1]) / 2])
        exten_vec = extend_vec_pt2_center - extend_vec_pt1_center
        k_extend = exten_vec[0] / exten_vec[1]
        b_extend_up = self.res_width - k_extend * left_boundarylineA_id[0]
        b_extend_dw = self.res_width - k_extend * left_boundarylineA_id[-1]

        # then it could be able to get the intersection point with boundary
        extend_ROI = np.array([
            np.array([self.res_width, left_boundarylineA_id[0]]),
            np.array([self.res_width, left_boundarylineA_id[-1]]),
            np.array([self.res_width + extend_dim,
                      int(((self.res_width + extend_dim) - b_extend_dw) / k_extend)]),
            np.array([self.res_width + extend_dim,
                      int(((self.res_width + extend_dim) - b_extend_up) / k_extend)])
        ])
        # pdb.set_trace()
        img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

        skeleton = skeletonize(img_thresh_extend)

        self.img_raw_skeleton = np.argwhere(skeleton[:, 0:self.res_width] == 1)

        ## display results
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
        # ax = axes.ravel()
        # ax[0].imshow(img_thresh_extend, cmap=plt.cm.gray)
        # ax[0].axis('off')
        # ax[0].set_title('original')
        # ax[1].imshow(skeleton, cmap=plt.cm.gray)
        # ax[1].scatter(self.img_raw_skeleton[:, 1], self.img_raw_skeleton[:, 0], marker='o', s=1)
        # ax[1].axis('off')
        # ax[1].set_title('skeleton')
        # # ax[1].plot(xxx[:, 1], xxx[:, 0], 'ro-', markersize=2, linewidth=1)
        # fig.tight_layout()
        # plt.show()
        # pdb.set_trace()

        # ##
        # '''
        # compute reference points for contour
        # '''
        # contours, hierarchy = cv2.findContours(self.img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # # print(len(contours))
        # self.contour = np.squeeze(contours[0], axis=1)
        # self.img_contour = np.zeros(self.raw_img.shape, dtype=np.uint8)

        # # cv2.drawContours(self.img_contour, [contours[0]], 0, (255, 255, 255), 2)
        # # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        # # ax = axes.ravel()
        # # ax[0].imshow(self.img_contour, cmap=cm.gray)
        # # ax[0].set_title('Image Contour')
        # # plt.show()
        # # pdb.set_trace()

        # # idx = np.where(contours[0][:, 0] == self.res_width - 1)
        # idx = np.argwhere(self.contour[:, 0] == self.res_width - 1)

        # ref_y1 = np.min(self.contour[idx, 1])
        # ref_y2 = np.max(self.contour[idx, 1])

        # ref_point1 = np.array([self.res_width - 1, ref_y1])
        # ref_point2 = np.array([self.res_width - 1, ref_y2])

        # self.boundary_pt_withimage1 = ref_point1
        # self.boundary_pt_withimage2 = ref_point2

        ref_point1 = None

        return ref_point1

    def getTipEllipseInterpolationPoint(self):

        # cx, cy, amaj, amin, angle
        xy_ellipse_last = []
        N = 10
        i = torch.arange(N)
        alpha = i * (2 * np.pi / (N - 1))

        # https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio

        x = self.list_of_ellipses[-1, 0] + self.list_of_ellipses[-1, 2] * torch.cos(alpha) * torch.cos(
            self.list_of_ellipses[-1, 4]) - self.list_of_ellipses[-1, 3] * torch.sin(alpha) * torch.sin(
                self.list_of_ellipses[-1, 4])
        y = self.list_of_ellipses[-1, 1] + self.list_of_ellipses[-1, 2] * torch.cos(alpha) * torch.sin(
            self.list_of_ellipses[-1, 4]) + self.list_of_ellipses[-1, 3] * torch.sin(alpha) * torch.cos(
                self.list_of_ellipses[-1, 4])
        # x = self.list_of_ellipses[-1, 0] + self.list_of_ellipses[-1, 2] * torch.cos(alpha)
        # y = self.list_of_ellipses[-1, 1] + self.list_of_ellipses[-1, 3] * torch.sin(alpha)
        # x = self.list_of_ellipses[6, 0] + self.list_of_ellipses[6, 2] * torch.cos(alpha)
        # y = self.list_of_ellipses[6, 1] + self.list_of_ellipses[6, 3] * torch.sin(alpha)

        candi_list = torch.cat((torch.unsqueeze(x, 1), torch.unsqueeze(y, 1)), dim=1)
        # print(xy_ellipse_last)
        candi_list = torch.cat(
            (candi_list,
             torch.unsqueeze(torch.stack((self.list_of_ellipses[-2, 0], self.list_of_ellipses[-2, 1]), dim=0), dim=0)),
            dim=0)
        # print(candi_list)

        # https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
        # sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
        sign = (self.list_of_inters_point2[-1, 0] -
                self.list_of_inters_point1[-1, 0]) * (candi_list[:, 1] - self.list_of_inters_point1[-1, 1]) - (
                    self.list_of_inters_point2[-1, 1] -
                    self.list_of_inters_point1[-1, 1]) * (candi_list[:, 0] - self.list_of_inters_point1[-1, 0])
        sign_compare = sign[-1]

        if sign_compare >= 0:
            sign_new = torch.where(sign >= 0., torch.tensor(0.0), sign)
        else:
            sign_new = torch.where(sign < 0., torch.tensor(0.0), sign)

        sign_new = torch.where(torch.abs(sign_new) <= 20.0, torch.tensor(0.0), sign)
        # print(sign)
        # print(sign_new)

        idx = torch.squeeze(torch.nonzero(sign_new), 1)
        # print(sign_new, '+++++')
        # print(torch.nonzero(sign_new), '+++++')

        xy_candi = candi_list[idx, :]

        # sort
        dis_xy_interp = torch.linalg.norm(xy_candi - self.list_of_inters_point1[-1, :], ord=2, dim=1)
        sorted_dis, sorted_idx = torch.sort(dis_xy_interp, dim=0)
        xy_tipEllipse_interp = xy_candi[sorted_idx, :]

        # print(candi_list)
        # print(xy_candi)
        # print(xy_tipEllipse_interp)

        # print(xy_tipEllipse_interp, '++++')

        # xy_elliplast_interp = candi_list[idx, :].detach().numpy()
        # xy = candi_list.detach().numpy()

        # # Generating figure
        # print(xy)
        # fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        # ax = axes.ravel()
        # ax[0].imshow(cv2.cvtColor(self.raw_img_rgb, cv2.COLOR_BGR2RGB))
        # ax[0].set_title('Initial Intersection Points (Only Shape Parameters)')
        # # ax[0].plot(xy[:, 0], xy[:, 1], 'go', markersize=0.2)
        # ax[0].plot(xy_tipEllipse_interp[:, 0].detach().numpy(), xy_tipEllipse_interp[:, 1].detach().numpy(), 'ro', markersize=0.5)
        # ax[0].plot(self.list_of_inters_point1[:, 0].detach().numpy(), self.list_of_inters_point1[:, 1].detach().numpy(), 'bo', markersize=0.5)
        # ax[0].plot(self.list_of_inters_point2[:, 0].detach().numpy(), self.list_of_inters_point2[:, 1].detach().numpy(), 'bo', markersize=0.5)
        # plt.tight_layout()
        # plt.show()

        assert not torch.any(torch.isnan(xy_tipEllipse_interp))

        return xy_tipEllipse_interp

    def getIntersPointDistanceToRef(self, ref_point):

        # need to determin before which is the top list
        # print(self.list_of_inters_point2)

        # modeled boundaries : projection
        line1 = torch.stack((self.list_of_inters_point1[0, :], self.last_notOnImage_point1), dim=0)
        line2 = torch.stack((self.list_of_inters_point2[0, :], self.last_notOnImage_point2), dim=0)
        line_boundary = torch.tensor([[self.res_width - 1, 0], [self.res_width - 1, self.res_height - 1]])
        inters_p1 = self.mathLineIntersection(line1, line_boundary)
        inters_p2 = self.mathLineIntersection(line2, line_boundary)

        if torch.equal(inters_p1, torch.tensor([9999., 9999.])):
            inters_p1 = self.list_of_inters_point1[0, :]
        if torch.equal(inters_p2, torch.tensor([9999., 9999.])):
            inters_p2 = self.list_of_inters_point2[0, :]

        xy_projbdry_interp = []
        N = 5
        step = (inters_p1 - inters_p2) / N
        for i in range(N + 1):
            temp = inters_p2 + i * step
            xy_projbdry_interp.append(temp)
        xy_projbdry_interp = torch.stack(xy_projbdry_interp)

        # modeled boundaries : tip ellipse
        self.xy_tipEllipse_interp = self.getTipEllipseInterpolationPoint()
        # xy_tipEllipse_interp = torch.tensor([[315.4665, 341.6405]])
        # xy_tipEllipse_interp = torch.unsqueeze(self.list_of_ellipses[-1, 0:2], 0)
        # print(xy_tipEllipse_interp)

        # combine together
        list2 = torch.flip(self.list_of_inters_point2, dims=[0])
        # combine_list = torch.cat((self.list_of_inters_point1, list2), dim=0)

        self.contour_model_combine = torch.cat((torch.unsqueeze(
            inters_p1, 0), self.list_of_inters_point1, self.xy_tipEllipse_interp, list2, xy_projbdry_interp),
                                               dim=0)
        self.contour_model_onlyedges = torch.cat(
            (self.list_of_inters_point1, list2, torch.unsqueeze(self.list_of_inters_point1[0, :], 0)), dim=0)

        # self.contour_model_combine = torch.cat((torch.unsqueeze(inters_p1, 0), self.list_of_inters_point1, list2, xy_projbdry_interp), dim=0)
        # self.contour_model_combine = torch.cat((torch.unsqueeze(inters_p1, 0), self.list_of_inters_point1, list2, torch.unsqueeze(inters_p2, 0), torch.unsqueeze(inters_p1, 0)), dim=0)
        # self.contour_model_combine = torch.cat((self.list_of_inters_point1, list2), dim=0)

        # print(self.contour_model_combine)

        # # Generating figure
        # --------------------------------
        # fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        # ax = axes.ravel()
        # ax[0].imshow(cv2.cvtColor(self.raw_img_rgb, cv2.COLOR_BGR2RGB))
        # ax[0].set_title('With Interpolation of Boundary')
        # ax[0].plot(self.contour_model_combine[:, 0].detach().numpy(),
        #            self.contour_model_combine[:, 1].detach().numpy(),
        #            'ro-',
        #            markersize=2,
        #            linewidth=0.6)
        # ax[0].plot(self.xy_tipEllipse_interp[:, 0].detach().numpy(),
        #            self.xy_tipEllipse_interp[:, 1].detach().numpy(),
        #            'go',
        #            markersize=2)
        # ax[0].plot(xy_projbdry_interp[:, 0].detach().numpy(),
        #            xy_projbdry_interp[:, 1].detach().numpy(),
        #            'bo',
        #            markersize=2)
        # ax[1].imshow(cv2.cvtColor(self.raw_img_rgb, cv2.COLOR_BGR2RGB))
        # ax[1].set_title('Only Edges')
        # ax[1].plot(self.contour_model_onlyedges[:, 0].detach().numpy(),
        #            self.contour_model_onlyedges[:, 1].detach().numpy(),
        #            'ro-',
        #            markersize=2,
        #            linewidth=0.6)
        # plt.tight_layout()
        # plt.show()

        fig = plt.figure(figsize=plt.figaspect(0.7))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cv2.cvtColor(self.raw_img_rgb, cv2.COLOR_BGR2RGB))
        ax.set_title('Contour with interpolation of boundary')
        ax.plot(self.contour_model_combine[:, 0].detach().numpy(),
                self.contour_model_combine[:, 1].detach().numpy(),
                'ro-',
                markersize=2,
                linewidth=0.6)
        ax.plot(self.xy_tipEllipse_interp[:, 0].detach().numpy(),
                self.xy_tipEllipse_interp[:, 1].detach().numpy(),
                'go',
                markersize=2)
        ax.plot(xy_projbdry_interp[:, 0].detach().numpy(),
                xy_projbdry_interp[:, 1].detach().numpy(),
                'bo',
                markersize=2)
        plt.tight_layout()

        # # Saving figure
        # --------------------------------
        fig.savefig(self.save_dir + '/contours/contour_img' + str(self.img_id) + '_itr_' + str(self.GD_Iteration) +
                    '.jpg')  # save the figure to file
        plt.close(fig)

        # combine_list = torch.cat((combine_list, ref_point), dim=0)

        # combine_list2 = np.delete(combine_list, 0, axis=0)
        # combine_list2 = np.vstack((combine_list2, self.list_of_inters_point1[0, :]))
        # dis_combine_list = np.linalg.norm(combine_list - combine_list2, ord=None, axis=1, keepdims=False)
        # dis_sum = np.cumsum(dis_combine_list)

        diff_list = torch.diff(self.contour_model_combine, dim=0)
        dis_diff = torch.linalg.norm(diff_list, ord=None, dim=1)
        dis_sum = torch.cumsum(dis_diff, dim=0)
        dis_sum = torch.cat((torch.tensor([0]), dis_sum), dim=0)

        dis_to_ref = dis_sum
        # dis_to_ref = torch.linalg.norm(combine_list[0, :] - ref_point, dim=1) + dis_sum
        dis_to_ref = dis_to_ref / dis_to_ref[-1]

        # print(np.shape(combine_list)[0])
        # print(np.shape(self.list_of_inters_point2)[0])
        # print(aaaaaaaaaaaaaa)
        # print(dis_to_ref)

        assert not torch.any(torch.isnan(dis_to_ref))

        return dis_to_ref

    def getContourMaskOut(self, ref_point):

        # dis_to_ref1 = np.linalg.norm(self.list_of_inters_point1 - ref_point, ord=None, axis=1, keepdims=False)
        # dis_to_ref2 = np.linalg.norm(self.list_of_inters_point2 - ref_point, ord=None, axis=1, keepdims=False)

        # print(self.list_of_inters_point1 - ref_point)
        # print(dis_to_ref1)
        # print(dis_to_ref2)

        contour = self.contour.copy()
        # print(contour)

        # curvature : https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        vel = np.gradient(contour, axis=0)
        vel_grad = np.gradient(vel, axis=0)
        ds_dt = np.linalg.norm(vel, axis=1)
        d2s_dt2 = np.gradient(ds_dt)

        tangent = np.array([1 / ds_dt, 1 / ds_dt]).transpose() * vel
        zero_vec = np.array([1, 0])
        curvature = np.abs(vel_grad[:, 0] * vel[:, 1] - vel[:, 0] * vel_grad[:, 1]) / (vel[:, 0]**2 + vel[:, 1]**2)**1.5

        # angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        angle = np.arctan2(np.linalg.norm(np.cross(tangent, zero_vec)), np.dot(tangent, zero_vec)) * 180.0 / np.pi
        angle_grad = np.abs(np.gradient(angle))

        # the value should be set manually
        # thre = np.argwhere(angle_grad > 0.45)
        # print(thre)

        # to_delete = np.hstack((np.arange(853, 936), np.arange(1716, np.shape(contour)[0])))
        to_delete = np.hstack((np.arange(359, 372), np.arange(708, np.shape(contour)[0])))
        contour_maskout = np.delete(contour, to_delete, axis=0)

        # contour_maskout = np.vstack((contour_maskout, ref_point))
        contour_maskout = np.vstack(
            (self.boundary_pt_withimage1, contour_maskout, self.boundary_pt_withimage2, self.boundary_pt_withimage1))

        # # reduce size
        # idx = np.round(np.linspace(0, len(contour_maskout) - 1, np.shape(dis_to_ref)[0] * 1)).astype(int)
        # print(idx)
        # contour_maskout = contour_maskout[idx, :]

        # fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        # # ax = axes.ravel()
        # axes[0].plot(angle, 'go-', linewidth=1, markersize=2)
        # axes[0].set_title('angle')
        # axes[1].plot(angle_grad, 'ro-', linewidth=1, markersize=2)
        # axes[1].set_title('angle gradient')
        # axes[2].plot(contour[:, 0], contour[:, 1], 'bo', linewidth=1, markersize=2)
        # axes[2].set_title('contour')
        # axes[2].set_xlim((0, self.raw_img.shape[1] + 100))
        # axes[2].set_ylim((self.raw_img.shape[0], 0))
        # axes[3].plot(contour[:, 0], contour[:, 1], 'b.', linewidth=1, markersize=1)
        # # axes[3].plot(contour_maskout[852:854, 0], contour_maskout[852:854, 1], 'go', linewidth=1, markersize=2)
        # axes[3].plot(contour_maskout[:, 0], contour_maskout[:, 1], 'go', linewidth=1, markersize=2)
        # axes[3].set_title('contour after masking out')
        # axes[3].set_xlim((0, self.raw_img.shape[1] + 100))
        # axes[3].set_ylim((self.raw_img.shape[0], 0))
        # plt.show()

        # # print(curvature)
        # # print(contour)
        # print(aaa)

        self.contour_maskout = contour_maskout

    def getContourByDistance(self, dis_to_ref, ref_point):

        # contour_maskout = torch.as_tensor(self.contour_maskout)
        contour_maskout = torch.as_tensor(self.contour).float()
        # print(contour_maskout)

        # method 1 :
        # contour2 = np.delete(contour_maskout, 0, axis=0)
        # contour2 = np.vstack((contour2, contour_maskout[0, :]))
        # dis = np.linalg.norm(contour_maskout - contour2, ord=None, axis=1, keepdims=False)
        # dis_sum = np.cumsum(dis)

        diff_contour = torch.diff(contour_maskout, axis=0)
        dis_diff = torch.linalg.norm(diff_contour, ord=None, axis=1)
        dis_sum = torch.cumsum(dis_diff, dim=0)
        dis_sum = torch.cat((torch.tensor([0]), dis_sum), dim=0)
        # print(dis_sum)

        # dis_contour = torch.linalg.norm(contour_maskout[0, :] - ref_point) + dis_sum
        dis_contour = dis_sum
        dis_contour = dis_contour / dis_contour[-1]

        # print(dis_contour)
        # print(dis_contour)
        # print(dis_to_ref)

        # print(np.shape(dis_contour)[0])
        # print(np.shape(contour_maskout)[0])
        # print(np.shape(dis_to_ref)[0])
        # print(dis_to_ref[25:30])
        # print(dis_to_ref[29] - dis_to_ref[28])
        # print(dis_to_ref[28])
        # print(dis_to_ref[29])
        # print(dis_contour[853] - dis_contour[852])
        # # print(dis_contour[850:860])
        # print(dis_contour[852])
        # print(dis_contour[853])

        # print(contour_maskout)
        # print(dis_to_ref)

        contour_by_dis = []
        for i in range(dis_to_ref.shape[0]):
            # for i in range(29):
            err = torch.abs(dis_contour - dis_to_ref[i])
            # print(err, '---', i)
            # print(dis_to_ref[i])
            index = torch.argmin(err)
            # print(index)

            temp = torch.stack([contour_maskout[index, 0], contour_maskout[index, 1]], dim=0)
            contour_by_dis.append(temp)

        # aaa = np.linalg.norm(contour_maskout[130:140, :] - ref_point, ord=None, axis=1, keepdims=False)
        # print(dis_contour[130:140])
        # print(self.contour)
        # print(self.contour)
        # self.contour_raw_bydis = torch.as_tensor(contour_by_dis)
        self.contour_raw_bydis = torch.stack(contour_by_dis)
        # print(self.contour_raw_bydis)

        # print(self.contour_raw_bydis.shape)

        # self.contour_maskout = contour_maskout
        # contour_dis_to_ref = np.linalg.norm(self.contour - ref_point, ord=None, axis=1, keepdims=False)

    def getFourierGGbar(self, order_N, dis_to_ref):

        G_bar_init = torch.empty([1, 4 * order_N + 2])
        for i in range(dis_to_ref.shape[0]):
            G = torch.eye(2)
            for j in range(order_N):
                Fj = torch.tensor([[torch.cos(j * dis_to_ref[i]),
                                    torch.sin(j * dis_to_ref[i]), 0, 0],
                                   [0, 0, torch.cos(j * dis_to_ref[i]),
                                    torch.sin(j * dis_to_ref[i])]])
                G = torch.cat((G, Fj), dim=1)
                # if j == 4:
                #     print(Fj)
                #     print(G)
                #     print(ccccc)
            G_bar_init = torch.cat((G_bar_init, G), dim=0)

        # G_bar = np.delete(G_bar, 0, axis=0)
        G_bar = G_bar_init[1:, :]

        U, s, V = torch.linalg.svd(torch.matmul(torch.transpose(G_bar, 0, 1), G_bar))
        # print('//////////////////////', s)
        # nozero_s_id = np.argwhere(s >= 0.00001).reshape(-1, 1)
        # iszero_s_id = np.argwhere(s < 0.00001).reshape(-1, 1)
        # s[nozero_s_id] = 1 / (s[nozero_s_id])
        # s[iszero_s_id] = 0

        # print(type(1 / s))
        s_new = torch.where(s >= 1e-5, 1.0 / s, torch.tensor([0.0]))
        # print(s)
        # print(s_new)
        # print('//////////////////////', s)
        inv_GG_bar = torch.transpose(V, 0, 1) @ torch.diag(s_new) @ torch.transpose(U, 0, 1)
        # inv_GG_bar = torch.matmul(torch.transpose(V, 0, 1), torch.matmul(torch.diag(s_new), torch.transpose(U, 0, 1)))

        # print(inv_GG_bar - inv_GG_bar2)
        # inv_GG_bar = np.transpose(V) @ np.diag(s_new) @ np.transpose(U)
        # print(self.hessian_all_stretch[0:1, 0:100], '+++++').

        # inv_GG_bar = torch.inverse(torch.matmul(torch.transpose(G_bar, 0, 1), G_bar))
        # print(inv_GG_bar)

        GG = torch.matmul(inv_GG_bar, torch.transpose(G_bar, 0, 1))
        # print(GG)

        return GG

    #################################################################################################################################
    #################################################################################################################################
    def getFourierShapeUseRawContourObj(self, order_N, dis_to_ref_model):

        mask_id = torch.arange(0, (self.contour.shape[0] - 1), 10).int()
        mask_id = torch.cat((mask_id, torch.tensor([self.contour.shape[0] - 1])), dim=0).int()
        contour_raw = torch.as_tensor(self.contour[mask_id, :])

        c_rawimg = torch.transpose(torch.reshape(contour_raw, (1, -1)), 0, 1).float()
        c_model = torch.transpose(torch.reshape(self.contour_model_combine, (1, -1)), 0, 1).float()

        diff_list_raw = torch.diff(contour_raw, dim=0).float()
        dis_diff_raw = torch.linalg.norm(diff_list_raw, ord=None, dim=1)
        dis_sum_raw = torch.cumsum(dis_diff_raw, dim=0)
        dis_sum_raw = torch.cat((torch.tensor([0]), dis_sum_raw), dim=0)
        dis_to_ref_raw = dis_sum_raw
        dis_to_ref_raw = dis_to_ref_raw / dis_to_ref_raw[-1]

        GG_raw = self.getFourierGGbar(order_N, dis_to_ref_raw)
        GG_model = self.getFourierGGbar(order_N, dis_to_ref_model)

        s_rawimg = torch.matmul(GG_raw, c_rawimg)
        s_model = torch.matmul(GG_model, c_model)

        s_rawimg_norm = s_rawimg / torch.linalg.norm(s_rawimg)
        s_model_norm = s_model / torch.linalg.norm(s_model)

        # print(s_model_norm)
        # print(s_rawimg_norm)

        # abs_cost = torch.abs(s_model - s_rawimg)
        abs_cost = torch.abs(s_model_norm - s_rawimg_norm)
        obj_J_FourierUseRawContour = torch.sum(torch.float_power(abs_cost, 2))

        # print(contour_raw)
        # print(G_bar_raw.shape)
        # print(GG_raw.shape)
        # print(GG_model.shape)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        # # ax = axes.ravel()
        # axes[0].plot(s_rawimg, 'go-', linewidth=1, markersize=4)
        # axes[0].plot(s_model, 'ro-', linewidth=1, markersize=4)
        # axes[0].set_title('22 shape parameters (order 5)')
        # plt.show()

        # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        # # ax = axes.ravel()
        # axes[0].plot(s_rawimg_norm.detach(), 'go-', linewidth=1, markersize=4)
        # axes[0].plot(s_model_norm.detach(), 'ro-', linewidth=1, markersize=4)
        # axes[0].set_title('22 shape parameters (order 5)')
        # plt.show()

        # print(c_rawimg)

        return obj_J_FourierUseRawContour

    def getFourierShapeObj(self, order_N, dis_to_ref):

        c_rawimg = torch.transpose(torch.reshape(self.contour_raw_bydis, (1, -1)), 0, 1).float()
        c_model = torch.transpose(torch.reshape(self.contour_model_combine, (1, -1)), 0, 1).float()

        # print(dis_to_ref)
        # print(ccccc)

        G_bar_init = torch.empty([1, 4 * order_N + 2])
        for i in range(self.contour_raw_bydis.shape[0]):
            G = torch.eye(2)
            for j in range(order_N):
                Fj = torch.tensor([[torch.cos(j * dis_to_ref[i]),
                                    torch.sin(j * dis_to_ref[i]), 0, 0],
                                   [0, 0, torch.cos(j * dis_to_ref[i]),
                                    torch.sin(j * dis_to_ref[i])]])
                G = torch.cat((G, Fj), dim=1)
                # if j == 4:
                #     print(Fj)
                #     print(G)
                #     print(ccccc)
            G_bar_init = torch.cat((G_bar_init, G), dim=0)

        # G_bar = np.delete(G_bar, 0, axis=0)
        G_bar = G_bar_init[1:, :]

        # print(c_model.shape)
        # print(G_bar.shape)

        # print(G_bar.shape)
        # print(ccc)

        # print(G_bar)
        # print(ccccc)
        # print(G_bar.shape)
        # print(type(c_model))

        # print(c_rawimg)
        # print(c_model)

        U, s, V = torch.linalg.svd(torch.matmul(torch.transpose(G_bar, 0, 1), G_bar))
        # print('//////////////////////', s)
        # nozero_s_id = np.argwhere(s >= 0.00001).reshape(-1, 1)
        # iszero_s_id = np.argwhere(s < 0.00001).reshape(-1, 1)
        # s[nozero_s_id] = 1 / (s[nozero_s_id])
        # s[iszero_s_id] = 0

        # print(type(1 / s))
        s_new = torch.where(s >= 1e-5, 1.0 / s, torch.tensor([0.0]))
        # print(s)
        # print(s_new)
        # print('//////////////////////', s)
        inv_GG_bar = torch.transpose(V, 0, 1) @ torch.diag(s_new) @ torch.transpose(U, 0, 1)
        # inv_GG_bar = torch.matmul(torch.transpose(V, 0, 1), torch.matmul(torch.diag(s_new), torch.transpose(U, 0, 1)))

        # print(inv_GG_bar - inv_GG_bar2)
        # inv_GG_bar = np.transpose(V) @ np.diag(s_new) @ np.transpose(U)
        # print(self.hessian_all_stretch[0:1, 0:100], '+++++').

        # inv_GG_bar = torch.inverse(torch.matmul(torch.transpose(G_bar, 0, 1), G_bar))
        # print(inv_GG_bar)

        GG = torch.matmul(inv_GG_bar, torch.transpose(G_bar, 0, 1))
        # print(GG)
        s_rawimg = torch.matmul(GG, c_rawimg)
        s_model = torch.matmul(GG, c_model)

        s_rawimg_norm = s_rawimg / torch.linalg.norm(s_rawimg)
        s_model_norm = s_model / torch.linalg.norm(s_model)

        # abs_cost = torch.abs(s_model - s_rawimg)
        abs_cost = torch.abs(s_model_norm - s_rawimg_norm)
        obj_J_Fourier = torch.sum(torch.float_power(abs_cost, 2))

        # print(s_rawimg)
        # print(s_model)
        # print(s_rawimg_norm)
        # print(s_model_norm)
        # print(aaa)
        # print(obj_J)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        # # ax = axes.ravel()
        # axes[0].plot(s_rawimg, 'go-', linewidth=1, markersize=4)
        # axes[0].plot(s_model, 'ro-', linewidth=1, markersize=4)
        # axes[0].set_title('22 shape parameters (order 5)')
        # plt.show()

        # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        # # ax = axes.ravel()
        # axes[0].plot(s_rawimg_norm.detach(), 'go-', linewidth=1, markersize=4)
        # axes[0].plot(s_model_norm.detach(), 'ro-', linewidth=1, markersize=4)
        # axes[0].set_title('22 shape parameters (order 5)')
        # plt.show()

        # print(c_rawimg)

        return obj_J_Fourier

    def getTipObj(self):

        # x_error_tipellipse = torch.abs(62.0 - self.xy_tipEllipse_interp[:, 0]) / self.res_width
        # y_error_tipellipse = torch.abs(404.0 - self.xy_tipEllipse_interp[:, 1]) / self.res_width

        # 0-62/404
        # 1-91/265
        # 2-90/163
        # 3-91/58
        # 4-214/86
        # 5-208/202

        skeleton = torch.as_tensor(self.img_raw_skeleton)

        x_error_tip1 = torch.abs(skeleton[0, 0] - self.list_of_ellipses[-1, 0]) / self.res_width
        y_error_tip1 = torch.abs(skeleton[0, 1] - self.list_of_ellipses[-1, 1]) / self.res_height

        # x_error_tipellipse = torch.abs(59.0 - self.xy_tipEllipse_interp[:, 0]) / 1
        # y_error_tipellipse = torch.abs(404.0 - self.xy_tipEllipse_interp[:, 1]) / 1
        # x_error_tip1 = torch.abs(59.0 - self.list_of_ellipses[-1, 0]) / 1
        # y_error_tip1 = torch.abs(404.0 - self.list_of_ellipses[-1, 1]) / 1

        # obj_J_tip = x_error_tip1 + y_error_tip1 + torch.sum(x_error_tipellipse) + torch.sum(y_error_tipellipse)
        # print(self.list_of_ellipses[-1, 0:2])
        obj_J_tip = (x_error_tip1 + y_error_tip1) * 1

        return obj_J_tip

    def getRightMostCenterObj(self):

        centerline = []
        for i in range(self.proj_bezier_img.shape[0]):
            if self.isPointInImage(self.proj_bezier_img[i, :], self.res_width, self.res_height):
                centerline.append(self.proj_bezier_img[i, :])

        centerline = torch.flip(centerline, dims=[0])

        # Move by shifted starting
        # centerline_shift = centerline - (centerline[0, :] - skeleton[0, :])
        centerline_shift = centerline
        # print(ellipses)

        skeleton = torch.as_tensor(self.img_raw_skeleton)

        x_error_rightMostCnt = torch.abs(skeleton[-1, 0] - centerline_shift[-1, 0]) / self.res_width
        y_error_rightMostCnt = torch.abs(skeleton[-1, 1] - centerline_shift[-1, 1]) / self.res_height

        obj_J_rightMostCnt = (x_error_rightMostCnt + y_error_rightMostCnt) * 1

        return obj_J_rightMostCnt

    def getEllipseSegmentsObj(self):

        ellipses = torch.flip(self.list_of_ellipses[:, 0:2], dims=[0])
        skeleton = torch.as_tensor(self.img_raw_skeleton)

        # Move by shifted starting
        ellipses_shift = ellipses - (ellipses[0, :] - skeleton[0, :])
        # print(ellipses)

        diff_skeleton = torch.diff(skeleton, axis=0)
        dis_diff_skeleton = torch.linalg.norm(diff_skeleton, ord=None, axis=1)
        dis_sum_skeleton = torch.cumsum(dis_diff_skeleton, dim=0)
        dis_sum_skeleton = torch.cat((torch.tensor([0]), dis_sum_skeleton), dim=0)

        diff_ellipses = torch.diff(ellipses_shift, axis=0)
        dis_diff_ellipses = torch.linalg.norm(diff_ellipses, ord=None, axis=1)
        dis_sum_ellipses = torch.cumsum(dis_diff_ellipses, dim=0)
        dis_sum_ellipses = torch.cat((torch.tensor([0]), dis_sum_ellipses), dim=0)

        skeleton_by_dis = []
        for i in range(dis_sum_ellipses.shape[0]):
            err = torch.abs(dis_sum_ellipses[i] - dis_sum_skeleton)
            index = torch.argmin(err)
            temp = skeleton[index, ]
            skeleton_by_dis.append(temp)
        skeleton_by_dis = torch.stack(skeleton_by_dis)
        # print(skeleton_by_dis)

        err_segments = torch.linalg.norm(ellipses - skeleton_by_dis, ord=None, axis=1) / self.res_width
        err_segments_sum = torch.sum(err_segments)

        # print(err_segments)

        return err_segments_sum

    def getCenterlineSegmentsObj(self):

        # centerline = []
        # for i in range(self.proj_bezier_img.shape[0]):
        #     if self.isPointInImage(self.proj_bezier_img[i, :], self.res_width, self.res_height):
        #         centerline.append(self.proj_bezier_img[i, :])

        # centerline = torch.stack(centerline)
        centerline = torch.clone(self.proj_bezier_img)
        centerline = torch.flip(centerline, dims=[0])

        skeleton = torch.as_tensor(self.img_raw_skeleton).float()
        skeleton = torch.flip(skeleton, dims=[1])

        # Move by shifted starting
        # centerline_shift = centerline - (centerline[0, :] - skeleton[0, :])
        centerline_shift = centerline
        # print(ellipses)

        # -------------------------------------------------------------------------------
        # #### get normlized distance vector regarding RAW skeleton
        # diff_skeleton = torch.diff(skeleton, axis=0)
        # dis_diff_skeleton = torch.linalg.norm(diff_skeleton, ord=None, axis=1)
        # dis_sum_skeleton = torch.cumsum(dis_diff_skeleton, dim=0)
        # dis_sum_skeleton = torch.cat((torch.tensor([0]), dis_sum_skeleton), dim=0)
        # dis_sum_skeleton = dis_sum_skeleton / dis_sum_skeleton[-1]

        # #### get normlized distance vector regarding CENTERLINE model
        # diff_centerline = torch.diff(centerline_shift, axis=0)
        # dis_diff_centerline = torch.linalg.norm(diff_centerline, ord=None, axis=1)
        # dis_sum_centerline = torch.cumsum(dis_diff_centerline, dim=0)
        # dis_sum_centerline = torch.cat((torch.tensor([0]), dis_sum_centerline), dim=0)
        # dis_sum_centerline = dis_sum_centerline / dis_sum_centerline[-1]

        # skeleton_by_dis = []
        # for i in range(dis_sum_centerline.shape[0]):
        #     err = torch.abs(dis_sum_centerline[i] - dis_sum_skeleton)
        #     index = torch.argmin(err)
        #     temp = skeleton[index, ]
        #     skeleton_by_dis.append(temp)
        # skeleton_by_dis = torch.stack(skeleton_by_dis)

        # err_centerline_by_dis = torch.linalg.norm(centerline - skeleton_by_dis, ord=None, axis=1) / self.res_width
        # err_centerline_sum_by_dis = torch.sum(err_centerline_by_dis)
        # -------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------
        skeleton_by_corresp = []
        for i in range(skeleton.shape[0]):
            err = torch.linalg.norm(skeleton[i] - centerline_shift, ord=None, axis=1)
            index = torch.argmin(err)
            temp = centerline_shift[index, ]
            skeleton_by_corresp.append(temp)
        skeleton_by_corresp = torch.stack(skeleton_by_corresp)

        self.CENTERLINE_SHAPE = skeleton.shape[0]
        # err_skeleton_by_corresp = torch.linalg.norm(skeleton - skeleton_by_corresp, ord=None, axis=1) / self.res_width
        err_skeleton_by_corresp = torch.linalg.norm(skeleton - skeleton_by_corresp, ord=None, axis=1) / 1.0
        err_skeleton_sum_by_corresp = torch.sum(err_skeleton_by_corresp) / self.CENTERLINE_SHAPE
        # -------------------------------------------------------------------------------

        err_obj_Tip = torch.linalg.norm(skeleton[0, :] - centerline[0, :], ord=None)

        # # https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        # id_select = torch.arange(0, self.CENTERLINE_SHAPE - 1, 10).long()

        # velocity_raw = torch.diff(skeleton[id_select, :], axis=0)
        # ds_dt_raw = torch.linalg.norm(velocity_raw, ord=None, axis=1)
        # tangent_raw = velocity_raw / ds_dt_raw.reshape((-1, 1))
        # tangent_raw_norm = tangent_raw / torch.linalg.norm(tangent_raw, ord=None, axis=1).reshape((-1, 1))
        # tangent_raw_diff = torch.diff(tangent_raw_norm, axis=0)
        # curvature_raw = torch.linalg.norm(tangent_raw_diff, ord=None, axis=1)
        # # normal_raw = tangent_raw_diff / curvature_raw.reshape((-1, 1))

        # velocity_opt = torch.diff(skeleton_by_corresp[id_select, :], axis=0)
        # ds_dt_opt = torch.linalg.norm(velocity_opt, ord=None, axis=1)
        # tangent_opt = velocity_opt / ds_dt_opt.reshape((-1, 1))
        # tangent_opt_norm = tangent_opt / torch.linalg.norm(tangent_opt, ord=None, axis=1).reshape((-1, 1))
        # tangent_opt_diff = torch.diff(tangent_opt_norm, axis=0)
        # curvature_opt = torch.linalg.norm(tangent_opt_diff, ord=None, axis=1)

        # # pdb.set_trace()

        # #### display results
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
        # ax = axes.ravel()
        # ax[0].plot(skeleton[:, 0].detach().numpy(), skeleton[:, 1].detach().numpy(), 'ro-', markersize=2, linewidth=1)
        # ax[0].plot(centerline[0:100, 0].detach().numpy(),
        #            centerline[0:100, 1].detach().numpy(),
        #            'bo-',
        #            markersize=2,
        #            linewidth=1)
        # # ax[0].plot(skeleton_by_dis[:, 0].detach().numpy(),
        # #            skeleton_by_dis[:, 1].detach().numpy(),
        # #            'go-',
        # #            markersize=2,
        # #            linewidth=1)

        # # ax[1].plot(centerline[:, 0].detach().numpy(),
        # #            centerline[:, 1].detach().numpy(),
        # #            'bo-',
        # #            markersize=2,
        # #            linewidth=1)
        # # ax[1].plot(skeleton_by_corresp[:, 0].detach().numpy(),
        # #            skeleton_by_corresp[:, 1].detach().numpy(),
        # #            'go-',
        # #            markersize=2,
        # #            linewidth=1)
        # fig.tight_layout()
        # plt.show()

        # pdb.set_trace()

        return err_skeleton_sum_by_corresp, err_obj_Tip

    def getCurveLengthObj(self, curve_length_gt):
        curve_3d = self.pos_bezier_cam
        diff_curve = torch.diff(curve_3d, axis=0)
        len_diff = torch.linalg.norm(diff_curve, ord=None, axis=1)
        len_sum = torch.sum(len_diff, dim=0)

        obj_J_curveLen = torch.abs(len_sum - curve_length_gt) * (1.0 / curve_length_gt)

        # pdb.set_trace()

        return obj_J_curveLen

    def getContourMoment(self, contour, x_i, y_j):

        xp = torch.pow(contour[:, 0], x_i)
        yp = torch.pow(contour[:, 1], y_j)

        moment = torch.sum(torch.mul(xp, yp) * 1)
        return moment

    def getContourMomentObj(self):

        inters_rawimg = self.contour_raw_bydis
        inters_model = self.contour_model_combine
        contour = torch.as_tensor(self.contour)

        M_raw10 = self.getContourMoment(contour, 1, 0)
        M_raw01 = self.getContourMoment(contour, 0, 1)
        M_raw00 = self.getContourMoment(contour, 0, 0)
        # M_raw10 = self.getContourMoment(inters_rawimg, 1, 0)
        # M_raw01 = self.getContourMoment(inters_rawimg, 0, 1)
        # M_raw00 = self.getContourMoment(inters_rawimg, 0, 0)
        cx_raw = M_raw10 / M_raw00
        cy_raw = M_raw01 / M_raw00

        M_model10 = self.getContourMoment(inters_model, 1, 0)
        M_model01 = self.getContourMoment(inters_model, 0, 1)
        M_model00 = self.getContourMoment(inters_model, 0, 0)
        cx_model = M_model10 / M_model00
        cy_model = M_model01 / M_model00

        # print(tip_id)
        # print(inters_rawimg[tip_id, :])
        # print(inters_model[tip_id, :])

        # print(cx_raw, cy_raw)
        # print(cx_model, cy_model)
        # print(aaa)

        # x_error = torch.abs(inters_rawimg[tip_id, 0] - inters_model[tip_id, 0]) / self.res_width
        # y_error = torch.abs(inters_rawimg[tip_id, 1] - inters_model[tip_id, 1]) / self.res_height

        x_error = torch.abs(cx_raw - cx_model) / self.res_width
        y_error = torch.abs(cy_raw - cy_model) / self.res_height

        # print(x_error)
        # print(y_error)

        obj_J_centroid = x_error + y_error

        return obj_J_centroid

    def getContourPolygonMomentObj(self):

        # BzrCURVE.plotOptimalIntersections(self.raw_img_rgb, self.contour_raw_bydis.detach().numpy(), self.contour_model_combine.detach().numpy())

        # inters_rawimg = self.contour_raw_bydis
        inters_rawimg = self.contour
        inters_model = self.contour_model_combine
        inters_onlyedges = self.contour_model_onlyedges

        # print(inters_rawimg)
        # print(inters_model)

        contour = torch.as_tensor(self.contour)

        img_polygon_model = torch.as_tensor(self.img_thresh.copy())
        img_polygon_onlyedges = torch.as_tensor(self.img_thresh.copy())
        img_polygon_raw = torch.as_tensor(self.img_thresh.copy())
        img_contour_raw = torch.as_tensor(self.img_thresh.copy())

        allimagepoints = torch.nonzero(img_polygon_model * 0 + 1)
        if_inside_model = self.rayTracingPolygon(allimagepoints, inters_model)
        if_inside_model = torch.reshape(if_inside_model, (int(self.res_height), -1))
        img_polygon_model = img_polygon_model * 0 + if_inside_model

        if_inside_onlyedges = self.rayTracingPolygon(allimagepoints, inters_onlyedges)
        if_inside_onlyedges = torch.reshape(if_inside_onlyedges, (int(self.res_height), -1))
        img_polygon_onlyedges = img_polygon_onlyedges * 0 + if_inside_onlyedges

        if_inside_raw = self.rayTracingPolygon(allimagepoints, inters_rawimg)
        if_inside_raw = torch.reshape(if_inside_raw, (int(self.res_height), -1))
        img_polygon_raw = img_polygon_raw * 0 + if_inside_raw

        # for i in range(int(self.res_width)):
        #     for j in range(int(self.res_height)):
        #         p = torch.tensor([i, j])
        #         if_inside_polygon = self.isInsidePolygon(inters_model, p)
        #         img_polygon_model[i, j] = if_inside_polygon * 1

        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # ax = axes.ravel()
        # ax[0].imshow(cv2.cvtColor(img_polygon_raw.numpy(), cv2.COLOR_BGR2RGB))
        # ax[0].set_title('img polygon raw')
        # ax[1].imshow(cv2.cvtColor(img_polygon_onlyedges.numpy(), cv2.COLOR_BGR2RGB))
        # ax[1].set_title('img polygon only edges')
        # ax[2].imshow(cv2.cvtColor(img_polygon_model.numpy(), cv2.COLOR_BGR2RGB))
        # ax[2].set_title('img polygon with bdry')
        # plt.tight_layout()
        # plt.show()

        poly_raw = torch.nonzero(img_polygon_raw)
        poly_model = torch.nonzero(img_polygon_model)

        M_raw10 = self.getContourMoment(poly_raw, 1, 0)
        M_raw01 = self.getContourMoment(poly_raw, 0, 1)
        M_raw00 = self.getContourMoment(poly_raw, 0, 0)
        cx_raw = M_raw10 / M_raw00
        cy_raw = M_raw01 / M_raw00

        M_model10 = self.getContourMoment(poly_model, 1, 0)
        M_model01 = self.getContourMoment(poly_model, 0, 1)
        M_model00 = self.getContourMoment(poly_model, 0, 0)
        cx_model = M_model10 / M_model00
        cy_model = M_model01 / M_model00

        x_error = torch.abs(cx_raw - cx_model) / self.res_width
        y_error = torch.abs(cy_raw - cy_model) / self.res_height

        # print(x_error)
        # print(y_error)

        obj_J_contourPolygon = x_error + y_error

        return obj_J_contourPolygon

    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################

    def mathLineIntersection(self, line1, line2):
        # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return torch.tensor([9999.0, 9999.0])
            # raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        inters = torch.stack((x, y), dim=0)

        return inters

    def rayTracingPolygon(self, points, poly):
        # points = points.detach().numpy()
        # poly = poly.detach().numpy()

        y, x = points[:, 0], points[:, 1]
        n = len(poly)
        inside = torch.zeros(points.shape[0])
        # inside = np.zeros(len(x))

        p2x = 0.0
        p2y = 0.0
        xints = 0.0
        p1x, p1y = poly[0, :]
        for i in range(n):
            p2x, p2y = poly[(i + 1) % n, :]

            # print(poly)
            # print(i)
            # print(p1x, p1y)
            # print(p2x, p2y)
            # print(min(p1y, p2y), max(p1y, p2y))

            idx = torch.nonzero((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)))
            # idx = np.nonzero((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)))

            # <-- Fixed here. If idx is null skip comparisons below.
            if len(idx):
                if p1y != p2y:
                    xints = (y[idx] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    # print(idx)
                    # print(xints)
                    # print(aaa)
                if p1x == p2x:
                    # print(idx.shape[0])
                    inside[idx] = torch.logical_not(inside[idx], out=inside[idx].to(torch.float))
                    # inside[idx] = ~inside[idx]
                else:
                    # print(xints)
                    # print(x[idx])
                    idxx = idx[x[idx] <= xints]
                    # inside[idxx] = torch.logical_not(inside[idxx], out=torch.empty(idxx.shape[0], dtype=torch.float))
                    inside[idxx] = torch.logical_not(inside[idxx], out=inside[idxx].to(torch.float))
                    # print(inside[idxx])
                    # inside[idxx] = ~inside[idxx]

            p1x, p1y = p2x, p2y
        return inside

    def ifOnSegmentPolygon(self, p, q, r):
        # Given three colinear points p, q, r,  the function checks if point q lies on line segment 'pr'

        if ((q[0] <= max(p[0], r[0])) & (q[0] >= min(p[0], r[0])) & (q[1] <= max(p[1], r[1])) &
            (q[1] >= min(p[1], r[1]))):
            return True

        return False

    def getOrientationPolygon(self, p, q, r):
        # To find orientation of ordered triplet (p, q, r).
        # The function returns following values
        # 0 --> p, q and r are colinear
        # 1 --> Clockwise
        # 2 --> Counterclockwise

        val = (((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))

        if val == 0:
            return 0  # Collinear
        if val > 0:
            return 1
        else:
            return 2  # Clock or counterclock

    def doIntersectPolygon(self, p1, q1, p2, q2):

        # Find the four orientations needed for general and special cases
        o1 = self.getOrientationPolygon(p1, q1, p2)
        o2 = self.getOrientationPolygon(p1, q1, q2)
        o3 = self.getOrientationPolygon(p2, q2, p1)
        o4 = self.getOrientationPolygon(p2, q2, q1)

        # General case
        if (o1 != o2) and (o3 != o4):
            return True

        # Special Cases
        # p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if (o1 == 0) and (self.ifOnSegmentPolygon(p1, p2, q1)):
            return True

        # p1, q1 and p2 are colinear and q2 lies on segment p1q1
        if (o2 == 0) and (self.ifOnSegmentPolygon(p1, q2, q1)):
            return True

        # p2, q2 and p1 are colinear and
        # p1 lies on segment p2q2
        if (o3 == 0) and (self.ifOnSegmentPolygon(p2, p1, q2)):
            return True

        # p2, q2 and q1 are colinear and
        # q1 lies on segment p2q2
        if (o4 == 0) and (self.ifOnSegmentPolygon(p2, q1, q2)):
            return True

        return False

    def isInsidePolygon(self, polygon, p):
        # Returns true if the point p lies inside the polygon[] with n vertices
        # https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/

        INT_MAX = self.res_width + 1

        n = polygon.shape[0]

        # There must be at least 3 vertices in polygon
        if n < 3:
            return 0

        # Create a point for line segment from p to infinite
        extreme_p = torch.tensor([INT_MAX, p[1]])
        count = i = 0

        while True:
            ip1 = (i + 1) % n

            # Check if the line segment from 'p' to 'extreme_p' intersects with the line egment from 'polygon[i]' to 'polygon[ip1]'
            if (self.doIntersectPolygon(polygon[i, :], polygon[ip1, :], p, extreme_p)):

                # If the point 'p' is colinear with line segment 'i-ip1', then check if it lies on segment. If it lies, return true, otherwise false
                if self.getOrientationPolygon(polygon[i], p, polygon[ip1]) == 0:
                    return self.ifOnSegmentPolygon(polygon[i], p, polygon[ip1])

                count += 1

            i = ip1

            if (i == 0):
                break

        # Return true if count is odd, false otherwise
        return (count % 2 == 1)

    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################

    def getProjCenterlineImage(self, proj_bezier_img):

        centerline_draw_img_rgb = self.raw_img_rgb.copy()

        # Draw centerline
        for i in range(proj_bezier_img.shape[0] - 1):
            # if not self.isPointInImage(proj_bezier_img[i, :], centerline_draw_img_rgb.shape[1], centerline_draw_img_rgb.shape[0]):
            #     continue
            # if not self.isPointInImage(proj_bezier_img[i + 1, :], centerline_draw_img_rgb.shape[1], centerline_draw_img_rgb.shape[0]):
            #     continue

            p1 = (int(proj_bezier_img[i, 0]), int(proj_bezier_img[i, 1]))
            p2 = (int(proj_bezier_img[i + 1, 0]), int(proj_bezier_img[i + 1, 1]))
            cv2.line(centerline_draw_img_rgb, p1, p2, (0, 100, 255), 1)

        # Draw tangent lines every few to check they are correct
        show_every_so_many_samples = 10
        l = 0.1
        tangent_draw_img_rgb = centerline_draw_img_rgb.copy()
        for i, p in enumerate(proj_bezier_img):
            if i % show_every_so_many_samples != 0:
                continue

            if not self.isPointInImage(p, tangent_draw_img_rgb.shape[1], tangent_draw_img_rgb.shape[0]):
                continue

            p_d = self.getProjPointCam(
                self.pos_bezier_cam[i] + l * self.der_bezier_cam[i] / torch.linalg.norm(self.der_bezier_cam[i]),
                self.cam_K)[0]

            if not self.isPointInImage(p_d, tangent_draw_img_rgb.shape[1], tangent_draw_img_rgb.shape[0]):
                continue

            # print('Out')
            tangent_draw_img_rgb = cv2.line(tangent_draw_img_rgb, (int(p[0]), int(p[1])), (int(p_d[0]), int(p_d[1])),
                                            (0.0, 0.0, 255.0), 1)

        # cv2.imwrite('./gradient_steps_imgs/centerline_draw_img_rgb_' + str(self.GD_Iteration) + '.jpg',
        #             centerline_draw_img_rgb)
        # cv2.imwrite('./gradient_steps_imgs/tangent_draw_img_rgb_' + str(self.GD_Iteration) + '.jpg', tangent_draw_img_rgb)

        return centerline_draw_img_rgb, tangent_draw_img_rgb

    def getProjPrimitivesImagePoster(self, proj_bezier_img):

        # Generating figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 4))
        ax = axes.ravel()

        proj_primitives_image = self.raw_img_rgb.copy()

        # plot cylinder
        line1 = self.list_of_edges1_ABC.detach().numpy()
        x1_line1 = np.zeros(line1.shape[0])
        y1_line1 = -(line1[:, 2] + line1[:, 0] * x1_line1) / line1[:, 1]
        x2_line1 = np.zeros(line1.shape[0]) + self.raw_img.shape[1]
        y2_line1 = -(line1[:, 2] + line1[:, 0] * x2_line1) / line1[:, 1]

        line2 = self.list_of_edges2_ABC.detach().numpy()
        x1_line2 = np.zeros(line2.shape[0])
        y1_line2 = -(line2[:, 2] + line2[:, 0] * x1_line2) / line2[:, 1]
        x2_line2 = np.zeros(line2.shape[0]) + self.raw_img.shape[1]
        y2_line2 = -(line2[:, 2] + line2[:, 0] * x2_line2) / line2[:, 1]
        for i in range(self.list_of_edges1_ABC.shape[0]):
            cv2.line(proj_primitives_image, (int(x1_line1[i]), int(y1_line1[i])), (int(x2_line1[i]), int(y2_line1[i])),
                     (0, 0, 230), 1)
            cv2.line(proj_primitives_image, (int(x1_line2[i]), int(y1_line2[i])), (int(x2_line2[i]), int(y2_line2[i])),
                     (230, 0, 0), 1)

        # Draw centerline
        ax[0].scatter(proj_bezier_img[:, 0].detach(), proj_bezier_img[:, 1].detach(), marker='o', s=1)

        ax[0].imshow(cv2.cvtColor(proj_primitives_image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Projection Cylinders')
        ax[0].set_xlim((0, self.raw_img.shape[1]))
        ax[0].set_ylim((self.raw_img.shape[0], 0))

        # plot circles
        for i in range(self.list_of_ellipses.shape[0]):
            draw_ellipese_v = self.list_of_ellipses[i, :].detach().numpy()

            cv2.ellipse(proj_primitives_image, (int(draw_ellipese_v[0]), int(draw_ellipese_v[1])),
                        (int(draw_ellipese_v[2]), int(draw_ellipese_v[3])),
                        draw_ellipese_v[4] * 180.0 / np.pi,
                        0,
                        360,
                        color=(0, 250, 0),
                        thickness=1)

        ax[1].scatter(self.list_of_inters_point1[:, 0].detach(),
                      self.list_of_inters_point1[:, 1].detach(),
                      marker='o',
                      s=8)
        ax[1].scatter(self.list_of_inters_point2[:, 0].detach(),
                      self.list_of_inters_point2[:, 1].detach(),
                      marker='o',
                      s=8)
        ax[1].imshow(cv2.cvtColor(proj_primitives_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Projection Circles')
        ax[1].set_xlim((0, self.raw_img.shape[1]))
        ax[1].set_ylim((self.raw_img.shape[0], 0))

        plt.tight_layout()

        # if self.GD_Iteration >= 2:
        #     plt.show()

        plt.close(fig)

        return proj_primitives_image

    def getProjPrimitivesImage(self, proj_bezier_img):

        # Generating figure
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        ax = axes.ravel()

        proj_primitives_image = self.raw_img_rgb.copy()

        # plot cylinder
        line1 = self.list_of_edges1_ABC.detach().numpy()
        x1_line1 = np.zeros(line1.shape[0])
        y1_line1 = -(line1[:, 2] + line1[:, 0] * x1_line1) / line1[:, 1]
        x2_line1 = np.zeros(line1.shape[0]) + self.raw_img.shape[1]
        y2_line1 = -(line1[:, 2] + line1[:, 0] * x2_line1) / line1[:, 1]

        line2 = self.list_of_edges2_ABC.detach().numpy()
        x1_line2 = np.zeros(line2.shape[0])
        y1_line2 = -(line2[:, 2] + line2[:, 0] * x1_line2) / line2[:, 1]
        x2_line2 = np.zeros(line2.shape[0]) + self.raw_img.shape[1]
        y2_line2 = -(line2[:, 2] + line2[:, 0] * x2_line2) / line2[:, 1]
        for i in range(self.list_of_edges1_ABC.shape[0]):
            cv2.line(proj_primitives_image, (int(x1_line1[i]), int(y1_line1[i])), (int(x2_line1[i]), int(y2_line1[i])),
                     (0, 0, 230), 1)
            # cv2.line(proj_primitives_image, (int(x1_line2[i]), int(y1_line2[i])), (int(x2_line2[i]), int(y2_line2[i])),
            #          (230, 0, 0), 1)

        # Draw centerline
        ax[0].scatter(proj_bezier_img[:, 0].detach(), proj_bezier_img[:, 1].detach(), marker='o', s=1)

        ax[0].imshow(cv2.cvtColor(proj_primitives_image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Projection Cylinders')
        ax[0].set_xlim((0, self.raw_img.shape[1]))
        ax[0].set_ylim((self.raw_img.shape[0], 0))

        # plot circles
        for i in range(self.list_of_ellipses.shape[0]):
            draw_ellipese_v = self.list_of_ellipses[i, :].detach().numpy()

            cv2.ellipse(proj_primitives_image, (int(draw_ellipese_v[0]), int(draw_ellipese_v[1])),
                        (int(draw_ellipese_v[2]), int(draw_ellipese_v[3])),
                        draw_ellipese_v[4] * 180.0 / np.pi,
                        0,
                        360,
                        color=(0, 250, 0),
                        thickness=1)

        ax[1].scatter(self.list_of_inters_point1[:, 0].detach(),
                      self.list_of_inters_point1[:, 1].detach(),
                      marker='o',
                      s=8)
        ax[1].scatter(self.list_of_inters_point2[:, 0].detach(),
                      self.list_of_inters_point2[:, 1].detach(),
                      marker='o',
                      s=8)
        ax[1].imshow(cv2.cvtColor(proj_primitives_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Projection Circles')
        ax[1].set_xlim((0, self.raw_img.shape[1]))
        ax[1].set_ylim((self.raw_img.shape[0], 0))

        curve_3d = self.pos_bezier_cam.detach().numpy()
        # print(curve_3d)
        C_EM0inCam = self.C_EM0inCam.detach().numpy()
        # print(C_EM0inCam)
        ax[2].remove()
        ax[2] = fig.add_subplot(1, 4, 3, projection='3d')
        ax[2].plot3D(curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2], 'gray', linestyle='-', linewidth=2)
        ax[2].scatter(curve_3d[-1, 0], curve_3d[-1, 1], curve_3d[-1, 2], marker='o', s=20)
        ax[2].scatter(0, 0, 0, marker='*', s=20)
        ax[2].plot3D([0.0, 0.01], [0, 0.0], [0.0, 0.0], 'red', linestyle='-', linewidth=2)
        ax[2].plot3D([0.0, 0.0], [0, 0.005], [0.0, 0.0], 'green', linestyle='-', linewidth=2)
        ax[2].plot3D([0.0, 0.0], [0, 0.0], [0.0, 0.04], 'blue', linestyle='-', linewidth=2)

        ax[2].scatter(C_EM0inCam[0, 0], C_EM0inCam[0, 1], C_EM0inCam[0, 2], marker='o', s=20)
        ax[2].plot3D([curve_3d[-1, 0], C_EM0inCam[0, 0]], [curve_3d[-1, 1], C_EM0inCam[0, 1]],
                     [curve_3d[-1, 2], C_EM0inCam[0, 2]],
                     'blue',
                     linestyle='-',
                     linewidth=1)

        ax[2].set_xlabel('X')
        ax[2].set_ylabel('Y')
        ax[2].set_zlabel('Z')
        ax[2].locator_params(nbins=4, axis='x')
        ax[2].locator_params(nbins=4, axis='y')
        ax[2].locator_params(nbins=4, axis='z')
        ax[2].view_init(-75, -90)

        ax[3].remove()
        ax[3] = fig.add_subplot(1, 4, 4, projection='3d')
        ax[3].plot3D(curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2], 'gray', linestyle='-', linewidth=2)
        ax[3].scatter(curve_3d[-1, 0], curve_3d[-1, 1], curve_3d[-1, 2], marker='o', s=20)
        ax[3].scatter(0, 0, 0, marker='*', s=20)
        ax[3].plot3D([0.0, 0.01], [0, 0.0], [0.0, 0.0], 'red', linestyle='-', linewidth=2)
        ax[3].plot3D([0.0, 0.0], [0, 0.005], [0.0, 0.0], 'green', linestyle='-', linewidth=2)
        ax[3].plot3D([0.0, 0.0], [0, 0.0], [0.0, 0.04], 'blue', linestyle='-', linewidth=2)
        ax[3].set_xlabel('X')
        ax[3].set_ylabel('Y')
        ax[3].set_zlabel('Z')
        ax[3].locator_params(nbins=4, axis='x')
        ax[3].locator_params(nbins=4, axis='y')
        ax[3].locator_params(nbins=4, axis='z')
        ax[3].view_init(0, -90)

        plt.tight_layout()

        fig.savefig(self.save_dir + '/centerline/centerline_draw_img_rgb_' + str(self.img_id) + '_itr_' +
                    str(self.GD_Iteration) + '.jpg')  # save the figure to file
        plt.close(fig)

        return proj_primitives_image

    def plotContourSamples(self):

        img_contour = self.edges_img.copy()
        ret, img_thresh = cv2.threshold(self.raw_img.copy(), 249, 255, cv2.THRESH_BINARY)
        img_thresh = cv2.bitwise_not(img_thresh)
        # img_thresh = cv2.adaptiveThreshold(self.raw_img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # print(self.raw_img)

        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        idx = np.where(contours[0][:, 0] == self.res_width - 1)
        idx = np.argwhere(contours[0][:, 0] == self.res_width - 1)

        print(contours[0])
        print(contours[0].shape)
        print(idx.shape)

        print("Number of Contours found = " + str(len(contours)))

        # Draw all contours
        # -1 signifies drawing all contours
        black_img_contour = img_thresh.copy() * 0
        # cv2.drawContours(black_img_contour, contours, -1, (255, 0, 0), 3)
        for i in range(contours[0].shape[0]):
            cv2.circle(black_img_contour, (contours[0][i][0][0], contours[0][i][0][1]),
                       radius=1,
                       color=(255, 255, 255),
                       thickness=-1)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        ax = axes.ravel()

        ax[0].imshow(img_thresh, cmap=cm.gray)
        ax[0].set_title('thresholding')

        ax[1].imshow(self.edges_img, cmap=cm.gray)
        ax[1].set_title('Edge image')

        ax[2].imshow(black_img_contour, cmap=cm.gray)
        ax[2].set_title('Contour image')

        plt.show()

    def plotPreProcess(self):
        # show raw image
        # ---------------
        # plt.figure(1)
        # plt.imshow(BzrCURVE.raw_img, cmap='gray')dynamic
        # plt.show()

        # show binary image
        # ---------------
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        ax = axes.ravel()

        ax[0].imshow(self.raw_img, cmap=cm.gray)
        ax[0].set_title('Input image')

        ax[1].imshow(self.blur_raw_img, cmap=cm.gray)
        ax[1].set_title('Blurred image')

        ax[2].imshow(self.edges_img, cmap=cm.gray)
        ax[2].set_title('Canny edges')
        ax[2].set_xlim((0, self.raw_img.shape[1]))
        ax[2].set_ylim((self.raw_img.shape[0], 0))

        # for a in ax:
        #     a.set_axis_off()
        # plt.tight_layout()
        plt.show()

    def plotProjCenterline(self):
        centerline_draw_img_rgb = self.raw_img_rgb.copy()
        curve_3D_opt = self.pos_bezier_3D.detach().numpy()
        curve_3D_gt = self.pos_bezier_3D_gt.detach().numpy()
        curve_3D_init = self.pos_bezier_3D_init.detach().numpy()

        # Draw centerline
        for i in range(self.proj_bezier_img.shape[0] - 1):
            # if not self.isPointInImage(proj_bezier_img[i, :], centerline_draw_img_rgb.shape[1], centerline_draw_img_rgb.shape[0]):
            #     continue
            # if not self.isPointInImage(proj_bezier_img[i + 1, :], centerline_draw_img_rgb.shape[1], centerline_draw_img_rgb.shape[0]):
            #     continue

            p1 = (int(self.proj_bezier_img[i, 0]), int(self.proj_bezier_img[i, 1]))
            p2 = (int(self.proj_bezier_img[i + 1, 0]), int(self.proj_bezier_img[i + 1, 1]))
            cv2.line(centerline_draw_img_rgb, p1, p2, (0, 100, 255), 4)

        # show
        # ---------------
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

        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax1.plot3D(curve_3D_init[:, 0],
                   curve_3D_init[:, 1],
                   curve_3D_init[:, 2],
                   color='#a64942',
                   linestyle='-',
                   linewidth=2)  ## red
        ax1.plot3D(curve_3D_gt[:, 0], curve_3D_gt[:, 1], curve_3D_gt[:, 2], color='#1f640a', linestyle='-',
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
        # ax1.scatter(0, 0, 0, marker='o', s=20)
        # ax[1].plot3D([0.0, 0.01], [0, 0.0], [0.0, 0.0], 'red', linestyle='-', linewidth=2)
        # ax[1].plot3D([0.0, 0.0], [0, 0.005], [0.0, 0.0], 'green', linestyle='-', linewidth=2)
        # ax[1].plot3D([0.0, 0.0], [0, 0.0], [0.0, 0.04], 'blue', linestyle='-', linewidth=2)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.locator_params(nbins=4, axis='x')
        ax1.locator_params(nbins=4, axis='y')
        ax1.locator_params(nbins=4, axis='z')
        ax1.view_init(22, -26)
        ax1.set_title('init/gt/opt : red/green/blue')

        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(self.saved_opt_history[1:, 0], color='#6F69AC', linestyle='-', linewidth=1)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Loss')

        # plt.tight_layout()
        plt.show()

    def plotProjCylinderLines(self, centerline_draw_img_rgb, tangent_draw_img_rgb, cylinder_draw_img_rgb):

        # show
        # ---------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes.ravel()

        ax[0].imshow(cv2.cvtColor(centerline_draw_img_rgb, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Projected centerline')

        ax[1].imshow(cv2.cvtColor(tangent_draw_img_rgb, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Projected tangents')

        ax[2].imshow(cv2.cvtColor(cylinder_draw_img_rgb, cv2.COLOR_BGR2RGB))
        ax[2].set_title('Projected cylinders')

        plt.tight_layout()
        plt.show()

    def plotProjCircles(self, cylinder_draw_img_rgb, circle_draw_img_rgb):
        # show
        # ---------------
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        ax = axes.ravel()

        ax[0].imshow(cylinder_draw_img_rgb)
        ax[0].set_title('Projected cylinders')

        ax[1].imshow(circle_draw_img_rgb)
        ax[1].set_title('Projected circles')

        plt.tight_layout()
        plt.show()

    def plotIntersectionPoints(self, circle_draw_img_rgb):

        # Generating figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        ax = axes.ravel()

        circles_draw_img_rgb_cp = circle_draw_img_rgb.copy()
        # plot cylinder as well
        # print(self.list_of_edges_1.shape[0])
        for i in range(self.list_of_edges_1.shape[0]):
            e_1 = self.list_of_edges_1[i, :]
            e_2 = self.list_of_edges_2[i, :]

            a = torch.cos(e_1[1])
            b = torch.sin(e_1[1])
            x0 = a * e_1[0]
            y0 = b * e_1[0]
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))

            cv2.line(circles_draw_img_rgb_cp, (x1, y1), (x2, y2), (230, 0, 0), 2)

            a = torch.cos(e_2[1])
            b = torch.sin(e_2[1])
            x0 = a * e_2[0]
            y0 = b * e_2[0]
            x1 = int(x0 + 5000 * (-b))
            y1 = int(y0 + 5000 * (a))
            x2 = int(x0 - 5000 * (-b))
            y2 = int(y0 - 5000 * (a))

            cv2.line(circles_draw_img_rgb_cp, (x1, y1), (x2, y2), (0, 0, 230), 2)

        ax[0].imshow(cv2.cvtColor(circles_draw_img_rgb_cp, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Projection')
        ax[0].set_xlim((0, self.raw_img.shape[1]))
        ax[0].set_ylim((self.raw_img.shape[0], 0))

        # ax[0].scatter(self.list_of_inters_point1[:, 0], self.list_of_inters_point1[:, 1])
        # ax[0].scatter(self.list_of_inters_point2[:, 0], self.list_of_inters_point2[:, 1])

        ax[1].imshow(cv2.cvtColor(circle_draw_img_rgb, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Intersection Points')

        print(self.list_of_inters_point1[:, 0])

        ax[1].scatter(self.list_of_inters_point1[:, 0].detach(), self.list_of_inters_point1[:, 1].detach())
        ax[1].scatter(self.list_of_inters_point2[:, 0].detach(), self.list_of_inters_point2[:, 1].detach())

        plt.tight_layout()
        plt.show()

    def plotContourByDistance(self):

        # Generating figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        ax = axes.ravel()

        ax[0].imshow(self.edges_img, cmap=cm.gray)
        ax[0].set_title('Canny edges')
        ax[0].set_xlim((0, self.raw_img.shape[1]))
        ax[0].set_ylim((self.raw_img.shape[0], 0))

        # ax[0].scatter(self.list_of_inters_point1[:, 0], self.list_of_inters_point1[:, 1], marker='o')
        # ax[0].scatter(self.list_of_inters_point2[:, 0], self.list_of_inters_point2[:, 1], marker='o')
        ax[0].plot(self.list_of_inters_point1[:, 0], self.list_of_inters_point1[:, 1], 'go', markersize=4)
        ax[0].plot(self.list_of_inters_point2[:, 0], self.list_of_inters_point2[:, 1], 'go', markersize=4)

        ax[1].imshow(self.edges_img, cmap=cm.gray)
        ax[1].set_title('Contour by distance')
        ax[1].set_xlim((0, self.raw_img.shape[1]))
        ax[1].set_ylim((self.raw_img.shape[0], 0))
        # ax[1].scatter(self.contour_raw_bydis[:, 0], self.contour_raw_bydis[:, 1], marker='o')
        ax[1].plot(self.list_of_inters_point1[:, 0], self.list_of_inters_point1[:, 1], 'go', markersize=4)
        ax[1].plot(self.list_of_inters_point2[:, 0], self.list_of_inters_point2[:, 1], 'go', markersize=4)
        ax[1].plot(self.contour_raw_bydis[:, 0], self.contour_raw_bydis[:, 1], 'ro', markersize=4)

        # print(np.shape(self.contour_raw_bydis)[0])

        plt.tight_layout()
        plt.show()

    def plotHoughTransformRawImage(self):
        threshold = 100.0

        # Classic straight-line Hough transform
        # Set a precision of 1.0 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(self.edges_img, theta=tested_angles)

        # h = np.flip(h, axis=0)

        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step), np.rad2deg(theta[-1] + angle_step), d[0] - d_step, d[-1] + d_step]

        ax[0].imshow(h.astype(float) / np.sum(h.astype(float)), cmap=cm.gray, extent=bounds)
        ax[0].set_title('Hough transform')
        ax[0].set_xlabel('Angles (degrees)')
        ax[0].set_ylabel('Distance (pixels)')
        ax[0].axis('image')
        ax[0].set_aspect(1.0 / 30.0)

        h_thresh = h.astype(float) * (h > threshold)
        ax[1].imshow(h_thresh / np.sum(h_thresh.astype(float)), cmap=cm.gray, extent=bounds)
        ax[1].set_title('Thresholded {} hough transform'.format(threshold))
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')
        ax[1].set_aspect(1.0 / 30.0)

        # h = np.flip(h, axis=0)
        ax[2].imshow(self.edges_img, cmap=cm.gray)
        origin = np.array((0, self.edges_img.shape[1]))

        # h_peak, theta_peak, d_peak = hough_line_peaks(h, theta, d, min_distance=0, min_angle=0, threshold=threshold)

        id_list = np.where(h_thresh != 0)
        h_peak = h[id_list]
        d_peak = d[id_list[0]]
        theta_peak = theta[id_list[1]]

        print(h_peak.shape)
        print(theta_peak)

        # print(id_d.shape)
        # print(h_peak)
        # print(h.shape)
        # print(h_peak.shape)
        # print(theta_peak.shape)
        # print(np.where(h_thresh.flatten() != 0)[0].shape)
        # print(np.where(h_thresh.flatten() == 0)[0].shape)
        # print(np.where(self.edges_img.flatten() != 0)[0].shape)
        # print(np.sum(h.astype(float)))

        for _, angle, dist in zip(*(h_peak, theta_peak, d_peak)):
            # y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            # ax[2].plot(origin, (y0, y1), '-r')
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2), color=(0.8, 0, 0), linewidth=1, linestyle='-')
        ax[2].set_xlim(origin)
        ax[2].set_ylim((self.edges_img.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Thresholded {} detected lines'.format(threshold))

        plt.tight_layout()
        plt.show()

    def plotHoughTransformModeledCurve(self):

        threshold = 50.0
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(self.edges_img, theta=tested_angles)

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step), np.rad2deg(theta[-1] + angle_step), d[0] - d_step, d[-1] + d_step]
        print(bounds)

        # Generating figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        ax = axes.ravel()

        ax[0].imshow(h.astype(float) / np.sum(h.astype(float)), cmap=cm.gray, extent=bounds)
        ax[0].set_title('Raw-image Hough transform')
        ax[0].set_xlabel('Angles (degrees)')
        ax[0].set_ylabel('Distance (pixels)')
        ax[0].set_aspect(1.0 / 30.0)

        h_thresh = h.astype(float) * (h > threshold)

        perc_thresh = np.sum(np.where(threshold > h, 0, h)) / np.sum(h.astype(float)) * 100
        ax[1].imshow(h_thresh.astype(float) / np.sum(h_thresh.astype(float)), cmap=cm.gray, extent=bounds)
        ax[1].set_title('Raw-image Hough transform thresholded\n removed {}%'.format(perc_thresh))
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].set_aspect(1.0 / 30.0)

        # centerline_draw_img_rgb = self.h_thresh.copy()

        # h_modelcurve = np.zeros(h_thresh.shape)

        ax[3].scatter(self.list_of_edges_1[:, 1] * 180.0 / np.pi, self.list_of_edges_1[:, 0])
        ax[3].scatter(self.list_of_edges_2[:, 1] * 180.0 / np.pi, self.list_of_edges_2[:, 0])
        ax[3].grid()
        ax[3].set_title("Modeled Hough transform (zoomed)")
        ax[3].set_xlabel('Angles (degrees)')
        ax[3].set_ylabel('Distance (pixels)')
        ax[3].set_aspect(1.0 / 30.0)

        # h_thresh = np.flip(h_thresh, axis=0)
        ax[2].imshow(h_thresh.astype(float) / np.sum(h_thresh.astype(float)), cmap=cm.gray, extent=bounds)
        ax[2].set_title('Raw-image Hough transform (zoomed)')
        ax[2].set_xlabel('Angles (degrees)')
        ax[2].set_ylabel('Distance (pixels)')
        ax[2].set_aspect(1.0 / 30.0)
        ax[2].set_xlim(ax[3].get_xlim())
        ax[2].set_ylim(ax[3].get_ylim())

        plt.tight_layout()
        plt.show()

    def plotOptimalIntersections(self, img_rgb, list_A, list_B, list_C, curve_raw, curve_optimized):

        # Generating figure
        # fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        # ax = axes.ravel()
        fig = plt.figure(figsize=plt.figaspect(0.22))
        # ax = axes.ravel()

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        ax.set_title('Initial Intersection Points (Only Shape Parameters)')
        ax.plot(list_A[:, 0], list_A[:, 1], 'ro-', markersize=3, linestyle='-', linewidth=1)
        ax.plot(list_B[:, 0], list_B[:, 1], 'bo-', markersize=3, linestyle='-', linewidth=1)
        # ax[0].plot(list_B[:, 0], list_B[:, 1], 'go', markersize=3, linestyle='-', linewidth=1)
        # ax[1].plot(self.contour_raw_bydis[:, 0], self.contour_raw_bydis[:, 1], 'ro', markersize=4)

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        ax.set_title('Optimized Intersection Points (Including Moment)')
        ax.plot(list_C[:, 0], list_C[:, 1], 'go', markersize=4)
        # ax[1].plot(self.contour_raw_bydis[:, 0], self.contour_raw_bydis[:, 1], 'ro', markersize=4)

        # Data for a three-dimensional line
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.plot3D(curve_raw[:, 0], curve_raw[:, 1], curve_raw[:, 2], 'gray', linestyle='-', linewidth=1)
        ax.plot3D(curve_optimized[:, 0],
                  curve_optimized[:, 1],
                  curve_optimized[:, 2],
                  'red',
                  linestyle='-',
                  linewidth=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        plt.show()

    def forwardProjection(self, control_pts):

        # step 1 : get curve
        self.getBezierCurve(control_pts)

        # step 2 : get projection of center line
        self.proj_bezier_img = self.getProjPointCam(self.pos_bezier_cam, self.cam_K)

        assert not torch.any(torch.isnan(self.proj_bezier_img))

        # print(self.pos_bezier_cam)
        # print(proj_bezier_img)

        # # step 3 : proj cylinder
        # cylinder_draw_img_rgb = self.getProjCylinderImage(self.proj_bezier_img)
        # centerline_draw_img_rgb, tangent_draw_img_rgb = self.getProjCenterlineImage(self.proj_bezier_img)
        # self.plotProjCylinderLines(centerline_draw_img_rgb, tangent_draw_img_rgb, cylinder_draw_img_rgb)

        # # step 4 : proj circles
        # circle_draw_img_rgb = self.getProjCirclesImage(self.proj_bezier_img)
        # # self.plotProjCircles(cylinder_draw_img_rgb, circle_draw_img_rgb)

        # # step 5 : get intersections
        # self.getProjIntersectionCircleCylinder()
        # # self.plotIntersectionPoints(circle_draw_img_rgb)

        # getProjPrimitivesImage = self.getProjPrimitivesImage(self.proj_bezier_img)
        # getProjPrimitivesImagePoster = self.getProjPrimitivesImagePoster(self.proj_bezier_img)
        # # cv2.imwrite('./gradient_steps_imgs/centerline_draw_img_rgb_' + str(self.GD_Iteration) + '.jpg',
        # #             getProjPrimitivesImage)

    def getCostFun(self, ref_point_contour):

        # GT values
        P1 = torch.zeros(3)
        P2 = torch.zeros(3)
        C = torch.zeros(3)
        P1[0], P1[1], P1[2] = self.para[0], self.para[1], self.para[2]
        C[0], C[1], C[2] = self.para[3], self.para[4], self.para[5]
        P2[0], P2[1], P2[2] = self.para[6], self.para[7], self.para[8]

        P1_EM0inCam = P1 + self.OFF_SET
        P2_EM0inCam = P2 + self.OFF_SET
        self.C_EM0inCam = C + self.OFF_SET

        # P1_EM0inCam = (self.optimal_R_EMraw2Cam @ torch.reshape(P1, (3, 1))) + self.optimal_t_EMraw2Cam
        # P2_EM0inCam = (self.optimal_R_EMraw2Cam @ torch.reshape(P2, (3, 1))) + self.optimal_t_EMraw2Cam
        # C_EM0inCam = (self.optimal_R_EMraw2Cam @ torch.reshape(C, (3, 1))) + self.optimal_t_EMraw2Cam
        # P1_EM0inCam = torch.squeeze(torch.reshape(P1_EM0inCam, (1, 3)), 0)
        # P2_EM0inCam = torch.squeeze(torch.reshape(P2_EM0inCam, (1, 3)), 0)
        # C_EM0inCam = torch.reshape(C_EM0inCam, (1, 3))
        # self.C_EM0inCam = C_EM0inCam

        P1p = 2 / 3 * self.C_EM0inCam + 1 / 3 * P1_EM0inCam
        P2p = 2 / 3 * self.C_EM0inCam + 1 / 3 * P2_EM0inCam
        control_pts = torch.vstack((P1_EM0inCam, P1p, P2_EM0inCam, P2p))

        # print("Control points : ")
        # print(control_pts)
        # print("\n")
        # print(aaa)

        # # print('----------- STEP 1 ------------------')
        BzrCURVE.forwardProjection(control_pts)

        # # # [STEP] :
        # # # # print('----------- STEP 2 ------------------')
        # dis_to_ref = BzrCURVE.getIntersPointDistanceToRef(ref_point_contour)

        # # # # [STEP] :
        # # # # print('----------- STEP 3 ------------------')
        # BzrCURVE.getContourByDistance(dis_to_ref, ref_point_contour)

        # # # # [STEP] :
        # # # print('----------- STEP 4 ------------------')
        # obj_J_fourier = BzrCURVE.getFourierShapeObj(self.Fourier_order_N, dis_to_ref)

        # # obj_J_fourier_useRaw = BzrCURVE.getFourierShapeUseRawContourObj(self.Fourier_order_N, dis_to_ref)

        # # # print('----------- STEP  ------------------')
        # obj_J_tip = BzrCURVE.getTipObj()

        # # # print('----------- STEP  ------------------')
        # # obj_J_rightMostCnt = BzrCURVE.getRightMostCenterObj()

        # # print('----------- STEP  ------------------')
        # obj_J_segments = BzrCURVE.getEllipseSegmentsObj()
        obj_J_centerline, obj_J_tip = BzrCURVE.getCenterlineSegmentsObj()

        # # print('----------- STEP  ------------------')
        obj_J_curveLength = BzrCURVE.getCurveLengthObj(self.curve_length_gt)

        # # print('----------- STEP  ------------------')
        # obj_J_contour = BzrCURVE.getContourMomentObj()

        # # print('----------- STEP  ------------------')
        # obj_J_polygon = BzrCURVE.getContourPolygonMomentObj()

        # obj_J = self.AAA
        # obj_J = obj_J_shape + obj_J_moment
        # obj_J = obj_J_fourier + obj_J_tip + obj_J_polygon
        # obj_J = obj_J_fourier_useRaw + obj_J_tip + obj_J_polygon
        # obj_J = obj_J_fourier_useRaw + obj_J_tip + obj_J_contour
        # obj_J = obj_J_fourier + obj_J_tip
        # obj_J = obj_J_fourier + obj_J_tip + obj_J_curveLength + obj_J_contour
        # obj_J = obj_J_fourier + obj_J_tip + obj_J_curveLength + obj_J_contour + obj_J_segments
        # obj_J = obj_J_fourier*self.CENTERLINE_SHAPE/10 + (obj_J_tip)* self.CENTERLINE_SHAPE + (obj_J_curveLength  + obj_J_contour)*self.CENTERLINE_SHAPE/2 + obj_J_segments
        # obj_J = (obj_J_tip)* self.CENTERLINE_SHAPE + (obj_J_curveLength  + obj_J_contour)*self.CENTERLINE_SHAPE/2 + obj_J_segments
        # obj_J = (obj_J_tip)* self.CENTERLINE_SHAPE / 2  + obj_J_segments
        # print(self.CENTERLINE_SHAPE)
        # print(obj_J_segments)
        # print(obj_J_tip* self.CENTERLINE_SHAPE / 2)
        # print(obj_J_curveLength* self.CENTERLINE_SHAPE)
        # print(aaa)
        # obj_J = obj_J_segments + obj_J_tip*self.CENTERLINE_SHAPE/5  + obj_J_curveLength  + obj_J_contour*self.CENTERLINE_SHAPE/10

        # obj_J = obj_J_segments / self.CENTERLINE_SHAPE + obj_J_tip + obj_J_curveLength  #(good)

        # obj_J =  obj_J_tip * self.CENTERLINE_SHAPE + obj_J_curveLength * self.CENTERLINE_SHAPE + obj_J_contour * self.CENTERLINE_SHAPE + obj_J_segments
        # obj_J = obj_J_fourier + obj_J_tip + obj_J_curveLength + obj_J_contour + obj_J_rightMostCnt
        # obj_J = obj_J_fourier + obj_J_curveLength + obj_J_contour + obj_J_segments
        # obj_J = obj_J_fourier + obj_J_polygon
        # obj_J = obj_J_fourier
        # obj_J = obj_J_fourier
        # obj_J = obj_J_tip
        # obj_J = self.AAA

        # obj_J = obj_J_fourier + obj_J_curveLength + obj_J_contour

        obj_J = obj_J_centerline * self.loss_weight[0] + obj_J_tip * self.loss_weight[
            1] + obj_J_curveLength * self.loss_weight[2]

        print('obj_J_all :', obj_J.detach().numpy())
        # print('obj_J     :',
        #       obj_J_centerline.detach().numpy(),
        #       obj_J_tip.detach().numpy(),
        #       obj_J_curveLength.detach().numpy())

        return obj_J

    def getOptimize(self, ref_point_contour):
        def closure():
            self.optimizer.zero_grad()
            self.loss = self.getCostFun(ref_point_contour)
            self.loss.backward()
            # print(self.para.grad)

            # torch.nn.utils.clip_grad_norm_(self.para, 10.0)

            # print(aaaa)
            return self.loss

        self.optimizer.zero_grad()
        loss_history = []
        # last_loss = self.getCostFun(ref_point_contour)  # current loss value
        last_loss = 99.0  # current loss value

        converge = False  # converge or not
        self.GD_Iteration = 0  # number of updates

        while not converge and self.GD_Iteration < self.total_itr:
            # while iteration < 100:
            # calculate gradient
            # self.optimizer.zero_grad()
            # self.getCostFun(ref_point_contour).backward()

            # print(self.para.grad)

            # update
            # P1 = torch.tensor([0.308185839843749, -0.0133945312499999 -0.009, -0.09387333984375 - 0.003])
            # P2 = 0.268829492187499, 0.02199853515625, -0.0596194335937499,
            # C    0.2720, -0.0302, -0.1196
            self.optimizer.step(closure)

            with torch.no_grad():
                # self.para[0] = torch.clamp(self.para[0], -0.20, 0.30)
                # self.para[1] = torch.clamp(self.para[1], -0.1, 0.1)
                # self.para[2] = torch.clamp(self.para[2], -0.2, 0.1)
                # self.para[3] = torch.clamp(self.para[3], -0.20, 0.30)
                # self.para[4] = torch.clamp(self.para[4], -0.1, 0.1)
                # self.para[5] = torch.clamp(self.para[5], -0.2, 0.1)
                delta1 = 0.01
                delta2 = 0.01
                delta3 = 0.01
                self.para[0] = torch.clamp(self.para[0], self.para_init[0] - delta1, self.para_init[0] + delta1)
                self.para[1] = torch.clamp(self.para[1], self.para_init[1] - delta1, self.para_init[2] + delta1)
                self.para[2] = torch.clamp(self.para[2], self.para_init[2] - delta1, self.para_init[2] + delta1)
            # if abs(new_loss) < 0.2 and (abs(new_loss - current_loss) < 1e-2):
            if (abs(self.loss - last_loss) < 1e-6):
                converge = True

            self.GD_Iteration += 1
            # print("Curr grad : ", self.para.grad)
            print("Curr para : ", self.para)
            print("------------------------ FINISH ", self.GD_Iteration, " ^_^ STEP ------------------------ \n")

            last_loss = torch.clone(self.loss)
            loss_history.append(last_loss)

            saved_value = np.hstack((last_loss.detach().numpy(), self.para.detach().numpy()))
            self.saved_opt_history = np.vstack((self.saved_opt_history, saved_value))

            # self.plotProjCenterline()

        # self.plotProjCenterline()

        print("Final --->", self.para)
        print("GT    --->", self.para_gt)
        print("Error --->", torch.abs(self.para - self.para_gt))
        # np.savetxt(self.save_dir + '/final_optimized_para.csv', self.saved_opt_history, delimiter=",")

        # # plt.plot(loss_history)
        # plt.plot(loss_history, marker='o', linestyle='-', linewidth=1, markersize=4)
        # plt.show()

        # return self.saved_opt_history, self.para

    def plotOptimizeResult(self, ref_point_contour):
        P1 = torch.tensor([0., 0., 0.])
        P1p = torch.tensor([-1.0, 0.0, 0.0])

        # P2 = torch.tensor([-2.4010, -0.5384, -1.1555])
        # P2 = torch.tensor([-2.5, -1.5, 0.0])
        # P2p = torch.tensor([-2.4566, -0.6231, -0.1220])
        P2 = torch.tensor([-2.5000, -0.4635, -1.4266])

        P2p = torch.tensor([-2.4000, -0.2163, -0.6657])
        # self.control_pts = torch.vstack((P1, P1p, P2, P2p))
        self.para = P2
        self.getCostFun(ref_point_contour)

        list_raw = self.contour_model_combine.detach().numpy()
        curve_raw = self.pos_bezier_3D.detach().numpy()

        # =================================================================================
        # =================================================================================

        P2 = torch.tensor([-5, -1.5, 0.0])
        # P2p = torch.tensor([-2.4000, -0.2163, -0.6657])
        self.para = P2
        self.getCostFun(ref_point_contour)

        list_initial = self.contour_model_combine.detach().numpy()

        # =================================================================================
        # =================================================================================

        # P2 = torch.tensor([-4.5447, 0.5100, -2.5016])
        P2 = torch.tensor([-4.6014, 0.1703, -2.5079])
        # P2p = torch.tensor([-2.9394, 0.5682, 0.0904])
        self.para = P2
        self.getCostFun(ref_point_contour)

        list_optimized = self.contour_model_combine.detach().numpy()
        curve_optimized = self.pos_bezier_3D.detach().numpy()

        self.plotOptimalIntersections(self.raw_img_rgb, list_raw, list_initial, list_optimized, curve_raw,
                                      curve_optimized)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='Path to endoscopic image of curve (png)')
    parser.add_argument('--specs-input', help='Path to input Bezier specs file (npy)')
    parser.add_argument('--specs-output', help='Path to output Bezier specs file (npy)')
    args = parser.parse_args()


    ##### ===========================================
    #       Initialization
    ##### ===========================================
    ###image path
    #img_path = "/home/fei/ARCLab-CCCatheter/data/rendered_images/dof2_64/dof2_c40_0.0005_-0.005_0.2_0.01.png"

    ### ground truth bezier curve length from P0 to P1
    curve_length_gt = 0.1906

    bezier_gt_specs = np.load(args.specs_input)

    ### ground truth bezier points : [P0, PC, P1]
    para_gt = torch.tensor(bezier_gt_specs.flatten(), dtype=torch.float)

    #para_gt = torch.tensor([0.02, 0.002, 0.0, 0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.19168896],
    #                       dtype=torch.float)

    ### initialized bezier points : [P0, PC, P1]
    para_init = torch.tensor(
        [0.015, 0.0, 0.0, 0.01957763, 0.00191553, 0.09690971, -0.03142124, -0.00828425, 0.18168159],
        dtype=torch.float,
        requires_grad=True)
    # para_init = torch.tensor([0.02, 0.002, 0.0, 0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.19168896],
    #                          dtype=torch.float,
    #                          requires_grad=True)

    ### initialized weights for three different loss : centerline/tip/curvelength
    loss_weight = torch.tensor([1.0, 1.0, 1.0])

    ### total itr for gradients optimization
    total_itr = 100

    ##### ===========================================
    #       Main reconstruction of bezier curve
    ##### ===========================================
    ### constructor of Class reconstructCurve
    BzrCURVE = reconstructCurve(args.img, curve_length_gt, para_gt, para_init, loss_weight, total_itr)

    ### do optimization
    BzrCURVE.getOptimize(None)

    

    ### plot the final results
    BzrCURVE.plotProjCenterline()

    ###  print final optimized parameters
    print('Optimized parameters', BzrCURVE.para)