import torch
import numpy as np
import os

from image_bezier_loss import Img_bezier_loss
from skeleton import Skeleton
from bezier import Bezier

import loss_3d
import plot


class reconstruct():
    def __init__(self, test_path, n, p0, para_list, weight, para_gt, itr):
        '''
        initialization of the bezier curve

        para_gt is only used for calculate error
        '''
        self.test_path = test_path

        img_dir_path = test_path + '/images'
        self.centerline_img = Skeleton(img_dir_path, n).skeleton
        self.n = n

        self.p0 = p0
        self.para_list = para_list
        self.para = para_list[n]

        self.weight = weight

        self.para_gt = para_gt

        self.itr = itr

        self.verbose = 1
        self.saved_opt_history = np.zeros((1, self.para.shape[0] + 1))



    def get_cost_fun_centerline(self):

        bzr_for_2d = Bezier(self.p0, self.para)
        loss_2d = Img_bezier_loss(self.centerline_img, bzr_for_2d, self.weight)
        loss_1 = loss_2d.loss
        loss_2 = loss_3d.loss_3d_2(self.p0, self.para_list, self.test_path, self.n)

        loss = loss_1 + loss_2 * 1000

        return loss



    def getOptimize(self):
        
        def closure():
            self.optimizer.zero_grad()
            self.loss = self.get_cost_fun_centerline()
            self.loss.backward(retain_graph=True)
            return self.loss

        self.optimizer = torch.optim.Adam([self.para], lr=1e-3)
        self.optimizer.zero_grad()
        loss_history = []

        last_loss = 99.0

        converge = False
        self.GD_Iteration = 0

        while not converge and self.GD_Iteration < self.itr:
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

        self.error = torch.abs(self.para - self.para_gt)
        print("Final --->", self.para)
        print("GT    --->", self.para_gt)
        print("Error --->", self.error)

        return self.saved_opt_history, self.para



if __name__ == '__main__':

    test_path = '/home/candice/Documents/ARCLab-CatheterControl/results/UN015/D00_0073'
    n = 6
    

    para_np = np.load(test_path+'/p3d_poses.npy')
    config = np.load(test_path+'/params.npy')

    # bezier points: P0
    p0 = torch.tensor([0.02, 0.002, 0.0])
    para_offset = torch.tensor([0, 0, 0.0001, 0, 0.0001, 0])

    para = torch.from_numpy(para_np)
    para_init = torch.flatten(para[0]) + para_offset
    para_gt = torch.flatten(para[n])
    para_op = para_init.clone().detach().requires_grad_(True)

    bezier_init = Bezier(p0, para_init)
    bezier_gt = Bezier(p0, para_gt)

    loss_weight = torch.tensor([100.0, 1000.0, 10.0, 100.0])

    total_itr = 100

    # ground truth bezier length from P0 to P1
    curve_length_gt = 0.1906

    # bezier points: P0
    p0 = torch.tensor([0.02, 0.002, 0.0])


    ##### ===========================================
    #       Main reconstruction of bezier curve
    ##### ===========================================
    # Constructor of Class Reconstruction

    para_list = [0]*n
    para_list[0] = para_init

    error_matrix = np.zeros((n-1,n-1))

    for i in range(1,n):

        para_gt = torch.flatten(para[i])
        bezier_gt = Bezier(p0, para_gt)

        para_list[i] = para_op
        BzrCURVE = reconstruct(test_path, i, p0, para_list, loss_weight, para_gt, total_itr)

        saved_opt_history, para_op = BzrCURVE.getOptimize()
        # para_clone = para_op.clone().detach().requires_grad_(True)


        bezier_optimized = Bezier(p0, para_op)

        for j in range(i):
            error_matrix[i-1,j] = np.linalg.norm((para_list[j+1] - torch.flatten(para[j+1])).detach().numpy())

        plot.plot_final_result(test_path, i, 1, bezier_optimized.p3d_from_bezier, bezier_gt.p3d_from_bezier, bezier_init.p3d_from_bezier, BzrCURVE.error, bezier_optimized.p2d_from_bezier, saved_opt_history)

        print(para_list)

    print(error_matrix)