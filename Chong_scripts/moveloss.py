import interspace_transforms as it

import torch 

class loss_group():
    def __init__(self, img_dir, para_dir, ):
        '''
        initialize the values should be used for loss

        '''
        self.img_dir = img_dir
        self.para_dir = para_dir

    def get 

    def loss_1(self, shift_list):
         '''
        return the loss between projected and actual tip point after a small shiff

        Variables:
            p_start ((3,1) tensor): the fixed start point of the Bezier Curve
            p_end ((3,1) tensor): the end point of reconstructed Bezier Curve with every iteration
        '''
        num_samples = 30
        num_shift = shift_list.size(0) * shift_list.size(0) - 1

        p3d_from_bezier = torch.clone(self.p3d_from_bezier)

        p3d_selected = torch.zeros(num_samples,3)
        for i in range(num_samples):
            p3d_selected[i,:] = p3d_from_bezier[int(i*len(p3d_from_bezier)/num_samples)]
        
        p_start = self.P0_gt
        p2d_shifted_list = torch.zeros(num_shift, num_samples, 2)

        for i,p3d in enumerate(p3d_selected):
            config = it.inverse_kinematic_3dof(p_start, p3d)
            config_u = it.para1_transform_para2(config[0], config[1], r)
            p3d_shifted = it.multiple_shift(p_start, config_u[0], config_u[1], config[2], r, self.shift_list)
            p2d_shifted_list[:,i,:] = self.getProjPointCam(p3d_shifted, self.cam_K)
            

        obj_moved_end = torch.sum(torch.linalg.norm((p2d_shifted_list - self.p2d_end_shift_gt), dim=0))

        return obj_moved_end

