import random
import numpy as np
import cv2

from cc_catheter import CCCatheter
import transforms
import bezier_interspace_transforms
from bezier_set import BezierSet

class DCCatheter(CCCatheter):

    def __init__(self, curvature_list, control_pts):
        '''
        Args:
            p_0 ((3,) numpy array): start point of catheter

        '''
        self.curvature_list = curvature_list
        self.control_pts = control_pts
        self.cccatheter_list = np.zeros(len(curvature_list))
        self.dc_pt_list = []
        self.target_dc_pt_list = []

    def set_separate_cccatheter(self, i, loss_2d, tip_loss, n_mid_points, n_iter, verbose=1):
        '''
        set one constant curvature curve in discrete curvature curve
        Args:
            i: the sign of constant curvature curve segmant
        '''
        p_0 = self.control_pts[i,:]
        r = 1/self.curvature_list[i]
        dist = np.linalg.norm(self.control_pts[i,:]-self.control_pts[i+1.:])
        theta = np.arccos((2*r**2-dist**2)/(2*r*r))
        l = theta*r
        self.cccatheter_list[i]=CCCatheter(p_0, l, r, loss_2d, tip_loss, n_mid_points, n_iter, verbose=1)

    def calculate_dc_points(self, target=False):
        '''
        Calculate the list of points on the discrete constant curvature curve 
            given the control points and the curvatures
        Args:
            target (bool): whether to use target parameters for transform
        '''
        if target:
            for i, curve in enumerate(self.cccatheter_list):
                curve.calculate_cc_points(current_iter=0, init=False, target=True)
                self.target_dc_pt_list=np.concatenate(self.target_dc_pt_list, curve.target_cc_pt_list)
        else:
            for i, curve in enumerate(self.cccatheter_list):
                curve.calculate_cc_points(current_iter=0, init=False, target=False)
                self.dc_pt_list = np.concatenate(self.dc_pt_list, curve.cc_pt_list)

    def set_weight_matrix(self, w1=0, w2=0):
        '''
        Set weight matrix. The weight matrix is a 2x2 diagonal matrix.
        The n-th diagonal term corresponds to the damping weight of the n-th DoF control feedback

        Args:
            w1 (float): 1st DoF damping weight
            w2 (float): 2nd DoF damping weight
        '''
        self.weight_matrix = np.eye(2)
        self.weight_matrix[0,0] = w1
        self.weight_matrix[1,1] = w2

    