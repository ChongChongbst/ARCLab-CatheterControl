import torch
from bezier import Bezier


class Bezier_bezier_loss():
    def __init__(self, p0, para_1, shift, para_2, num_samples=30):
        '''
        Initialize the two Bezier Curve used for loss calculation
        '''
        self.bzr_1 = Bezier(p0, para_1, num_samples)
        self.bzr_1.apply_shift(shift)
        self.bzr_2 = Bezier(p0, para_2, num_samples)

        self.p3d_from_bzr_1 = self.bzr_1.p3d_from_bezier
        self.p3d_from_bzr_2 = self.bzr_2.p3d_from_bezier

        self.loss = self.calculate_loss()

    def calculate_loss(self):
        '''
        calculate the loss
        '''
        loss = torch.sum(torch.linalg.norm(self.p3d_from_bzr_1 - self.p3d_from_bzr_2))

        return loss


    
