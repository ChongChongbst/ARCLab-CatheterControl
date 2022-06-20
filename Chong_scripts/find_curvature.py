from turtle import end_fill
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import path_settings
import os
import sys
sys.path.append(path_settings.scripts_dir)

import YinFei_scripts.transforms as transforms
import YinFei_scripts.camera_settings as camera_settings
from linear_least_square_fit import ComputeCurvature



class FindCurvature:
    def __init__(self, p0, para):
        '''
        Args:
            para: the second and third control points of bezier curve
        '''

        p_0 = p0.detach().numpy()
        print(p_0)
        p3d = para.detach().numpy()
        p_mid = p3d[0:3]
        p_end = p3d[3:6]
        print(p_mid)
        self.cc_pt_list = np.stack((p_0, p_mid, p_end))
        self.bezier_specs = self.calculate_bezier_specs()

        self.camera_matrix = np.array([[883.00220751, 0, 320, 0],
                                [0, 883.00220751, 240, 0],
                                [0, 0, 1, 0]])

        self.width = 640
        self.height = 480

    def findcurvature(self, P):
        """
        Convert 3D constant curvature curve to 2D image frame, method 1 in paper

        Args:
            P((3,3) numpy array): a set of 3 points in 3D
                (1) one start point: one inflection point rigidly attached to the junction of adjacent segments
                (2) two points randomly selected along the curve
            camera_matrix((3,4) numpy array): projection matrix
        """
        A = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        P1_h = np.append(P[0,:], [1])
        P2_h = np.append(P[1,:], [1])
        P3_h = np.append(P[2,:], [1])

        m3 = self.camera_matrix[2,:]

        z1 = m3 @ P1_h.T
        z2 = m3 @ P2_h.T
        z3 = m3 @ P3_h.T

        if z1 == 0:
            z1 = 1e-10
        if z2 == 0:
            z2 = 1e-10
        if z3 == 0:
            z3 = 1e-10

        g1 = z2 * self.camera_matrix @ P1_h.T - z1 * self.camera_matrix @ P2_h.T
        g2 = z3 * self.camera_matrix @ P2_h.T - z2 * self.camera_matrix @ P3_h.T
        g3 = z3 * self.camera_matrix @ P1_h.T - z1 * self.camera_matrix @ P3_h.T

        p1_h = self.camera_matrix @ P1_h.T / z1
        p2_h = self.camera_matrix @ P2_h.T / z2
        p3_h = self.camera_matrix @ P3_h.T / z3

        n1 = (p1_h-p2_h).T/np.linalg.norm(p1_h-p2_h)
        n2 = (p2_h-p3_h).T/np.linalg.norm(p2_h-p3_h)
        n3 = (p1_h-p3_h).T/np.linalg.norm(p1_h-p3_h)

        w = (2 * z2 * z3 * (g3.T@A@g1))/((n1@g1) * (n2@g2) * (n3@g3))
        #w = (2 * z1 * z1* z2 * z3 * (g3.T@A@g1))/((n1@g1) * (n2@g2) * (n3@g3))

        return w


    def get_k_from_2d(self, p2d_list):
        """
        Calculate curvature on 2D image, method 2 in paper (least_square)
        
        Args:
            p2d_list ((3,2) numpy array): a set of 3 points in 2D
        
        """
        A = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

        p1 = np.append(p2d_list[0], [1])
        p2 = np.append(p2d_list[1], [1])
        p3 = np.append(p2d_list[2], [1])

        d1 = np.linalg.norm(p1-p2)
        d2 = np.linalg.norm(p2-p3)
        d3 = np.linalg.norm(p1-p3)

        s = 0.5*(p1-p3).T@A@(p1-p2)

        w = 4*s/(d1*d2*d3)

        return w
        

    def calculate_bezier_specs(self):
        """
        Calculate the Quadratic Bezier Specs for the current list of points on the constant curvature curve.
        Then convert it to Cubic Bezier Specs to suit Blender

        Args:
            cc_pt_list((3,3) numpy array): start point, middle point and end point on the constant curvature curve

        Returns:
            self.bezier_specs ((4, 3) numpy array): 
                the 1st row is the start point of Bezier curve;
                the 2nd row is the end point of Bezier curve;
                the 3rd row is the 1 control point of Bezier curve
                the 4th row is the 2 control point of Bezier curve

        Note:
            This only works for n_mid_points = 1
        """
        if len(self.cc_pt_list) != 3:
            print('[ERROR] Reconstruction is not compatible with more than 1 mid points')
            exit()

        p_0 = self.cc_pt_list[0]
        p_mid = self.cc_pt_list[1]
        p_end = self.cc_pt_list[2]

        bezier_specs = np.zeros((4, 3))

        c = (p_mid - (p_0 / 4) - (p_end / 4)) * 2

        c1 = (2*c + p_0)/3
        c2 = (2*c + p_end)/3

        bezier_specs[0, :] = p_0
        bezier_specs[1, :] = p_end
        bezier_specs[2, :] = c1
        bezier_specs[3, :] = c2

        return bezier_specs



    def get_points_on_curve(self, N):
        '''
        get N 3D points on the quadratic bezier curve 

        Args:
            N (int): the number of points got from the bezier curve
        '''
        p_0 = self.bezier_specs[0, :]
        p_end = self.bezier_specs[1, :]
        c1 = self.bezier_specs[2, :]
        c = (3*c1 -p_0)/2

        p3d_list = np.zeros((N,3))
        ts = np.linspace(0, 1, num=N)

        for i in range(N):
            t = ts[i]
            p3d_list[i] = c + (1-t)*(1-t)*(p_0-c) + t*t*(p_end-c)

        #self.p3d_list = self.p3d_list[10:]
        return p3d_list



    def project_3d_to_2d(self, p3d_list):
        '''
        Convert a set of 3D points in world frame to 2D points in image frame

        Args:
            p3d_list((,3) numpy array): a set of 3D points in world frame

        '''
        length = len(p3d_list)
        p2d_list = np.zeros((length, 2))
        for i in range(length):
            p2d_list[i] = transforms.world_to_image_transform(p3d_list[i], camera_settings.extrinsics, camera_settings.a, camera_settings.b,  camera_settings.center_x, camera_settings.center_y)
            p2d_list[i][0] = 640-p2d_list[i][0]
        return p2d_list


    def project_3d_camera_matrix(self, p3d_list):
        '''
        Use camera matrix to deal with the projection

        Args:
            p3d_list((,3) numpy array): a set of 3D points in world frame
            camera_matrix((3,4) numpy array): a camera matrix used to project

        '''
        length = len(p3d_list)
        p2d_list = np.zeros((length, 2))
        for i in range(length):
            p3d = np.append(p3d_list[i], [1])
            p2d = self.camera_matrix @ p3d
            p2d_list[i] = np.true_divide(p2d[:2], p2d[-1])
        return p2d_list



    def calculate_2d_curve_length(self):
        """
        get curve length on 2D image frame

        Args:
            bezier_specs((4,3) numpy array): the cubic bezier specs
        """
        p3d_5000 = self.get_points_on_curve(5000)
        p2d_5000 = self.project_3d_to_2d(p3d_5000)
        l = 0
        for i in range(len(p2d_5000)-1):
            dl = np.linalg.norm(p2d_5000[i]-p2d_5000[i+1])
            l = l + dl

        return l



    def get_TNB_curvature(self, pd_list, l):
        """
        Use TNB to calculate curvature

        Args:
            pd_list ((,2) numpy array): a set of 2D points in image frame

        """
        ds = l
        dT = pd_list[2]-pd_list[0]
        length = np.linalg.norm(dT)
        udT = dT/length
        k = np.linalg.norm(udT/ds)

        return k



    def curvature_change(self, pd_list, N, method):
        """
        get the curvature change along the image curve

        Args:
            pd_list in method 2 ((,2) numpy array): a set of 2D points in image frame
                    in method 1 ((,3) numpy array): a set of 3D points 
            N (int): the interval of the point
            method (int): the number of method used
                1. method from paper of Fan Xu
                2. least square method
                3. TNB method
        """
        length = len(pd_list)
        k = np.zeros(length-2)
        if method == 1:
            p3d_3 = np.zeros((3,3))

        else:
            p2d_3 = np.zeros((3,2))


        for i in range(length-2)[1:]:

            if method == 1:
                # method from the paper
                if (i-N) < 0:
                    p3d_3[0] = pd_list[0]
                else:
                    p3d_3[0] = pd_list[i-N]

                p3d_3[1] = pd_list[i]

                if (i+N) > (length-1):
                    p3d_3[2] = pd_list[length-1]
                else:
                    p3d_3[2] = pd_list[i+N]

                k[i] = self.findcurvature(p3d_3)

            else:
                # least square
                if (i-N) < 0:
                    p2d_3[0] = pd_list[0]
                else:
                    p2d_3[0] = pd_list[i-N]

                p2d_3[1] = pd_list[i]

                if (i+N) > (length-1):
                    p2d_3[2] = pd_list[length-1]
                else :
                    p2d_3[2] = pd_list[i+N]

                if method == 2:
                    x = p2d_3[:,0].T
                    y = p2d_3[:,1].T
                    comp_curv = ComputeCurvature()
                    k[i] = comp_curv.fit(x, y)              
                else :
                    # TNB
                    l2d = self.calculate_2d_curve_length(self.bezier_specs)
                    k[i] = self.get_TNB_curvature(p2d_3, l2d)

        return k

    def is_points_in_image(self, p_proj):
        '''
        Check whether a 2d point is in image or not

        Args:
            p_proj ((,2) numpy array): a list of 2D points on image frame
            width (int): the width of the image
            height (int): the height of the image
        '''
        p_image = []
        for p in p_proj:
            if p[0] < 0 or p[1] < 0 or p[0] > self.width or p[1] > self.height:
                continue
            p_image.append(p)
        return p_image

    def find_points_in_image(self, n_target, width, height):
        '''
        Find N points in image

        Args:
            N (int): the number of points should be found in image
            width (int): the width of the image
            height (int): the height of the image
        '''
        N = n_target
        p3d_temp = self.get_points_on_curve(N)
        p2d_temp = self.project_3d_to_2d(p3d_temp)
        p2d = self.is_points_in_image(p2d_temp, self.width, self.height)
        n = len(p2d)
        N = int(N*N/n)
        p3d_temp = self.get_points_on_curve(N)
        p2d_temp = self.project_3d_to_2d(p3d_temp)
        p2d = self.is_points_in_image(p2d_temp, self.width, self.height)
        n = len(p2d)

        while True:
            if n < n_target:
                N = N + 1
                p2d_temp = self.get_points_on_curve(N)
                p2d = self.is_points_in_image(p2d_temp, self.width, self.height)
                n = len(p2d)
            elif n > n_target:
                N = N - 1
                p2d_temp = self.get_points_on_curve(N)
                p2d = self.is_points_in_image(p2d_temp, self.width, self.height)
                n = len(p2d)
            else:
                break
        return p2d
            

  
##### ===========================================
#       Test Code For Chong
##### =========================================== 
    def test(self):

        path = "/home/candice/Documents/ARCLab-CatheterControl/results/UN015/D00_0021/" 
        img = cv2.imread(os.path.join(path, "images/blenderpic1.png"))
        p3d = np.load(os.path.join(path, "p3d_poses.npy"))

        p_0 = np.array([2e-2, 2e-3, 0])
        p_mid = p3d[-1, 0, :]
        p_end = p3d[-1, 1, :]
        cc_pt_list = np.stack((p_0, p_mid, p_end))
        para = torch.tensor(np.concatenate((p_mid, p_end)))

        camera_matrix = np.array([[883.00220751, 0, -320, 0],
                                [0, -883.00220751, -240, 0],
                                [0, 0, -1, 0]])


        find_k = FindCurvature(para)
        print("Cubic bezier specs are:")
        print(find_k.bezier_specs)

        # see if the camera projection is right
        p3d_list = find_k.get_points_on_curve(1000)
        p2d_list = find_k.project_3d_to_2d(p3d_list)
        p2d_list_cm = find_k.project_3d_camera_matrix(p3d_list)

        # # verify discrete centerline points with camera matrix projection
        # for i in range(1000):
        #     if not(np.isinf(p2d_list_cm[i,0]) or np.isinf(p2d_list_cm[i,1])):
        #         img = cv2.circle(img, (640-int(p2d_list_cm[i,0]),int(p2d_list_cm[i,1])), radius=5, color=(255, 0, 0), thickness=-1)

        # cv2.imwrite(path+"cm_proj_1.png", img)

        # verify discrete centerline points with verified projection
        # for i in range(1000):
        #     if not(np.isinf(p2d_list[i,0]) or np.isinf(p2d_list[i,1])):
        #         img = cv2.circle(img, (640-int(p2d_list[i,0]),int(p2d_list[i,1])), radius=5, color=(0, 255, 0), thickness=-1)

        # cv2.imwrite(path+"proj_1.jpg", img)


        # verify curvature calculated
        # p_0_substitute = p3d_list[20]
        # new_cc_pts = np.stack((p_0_substitute, p_mid, p_end))
        # w1, p2d = cc_3d_to_2d(new_cc_pts, camera_matrix)
        # print("constant curvature value 1:")
        # print(w1)
        #
        # p2d_3 = np.zeros((3,2))
        # p2d_3[0] = p2d_list[499]
        # p2d_3[1] = p2d_list[500]
        # p2d_3[2] = p2d_list[501]
        # w2 = get_k_from_curve(p2d_3)
        # print("constant curvature value 2:")
        # print(w2)
        #
        # l2d = calculate_2d_curve_length(bezier_specs)
        # k = get_TNB_curvature(p2d_3, l2d)
        # print("curvature calculated from TNB")
        # print(k)

        k1 = find_k.curvature_change(p3d_list[10:], 10, 1)
        x = np.linspace(0, len(k1), len(k1))
        plt.plot(x, k1)
        plt.xlabel('x')
        plt.ylabel('k')
        plt.title('curvature from method 1')
        plt.show()

        k2 = find_k.curvature_change(p2d_list[10:], 10, 2)
        x = np.linspace(0, len(k2), len(k2))
        plt.figure()
        plt.plot(x, k2)
        plt.xlabel('x')
        plt.ylabel('k')
        plt.title('curvature from least square')
        plt.show()