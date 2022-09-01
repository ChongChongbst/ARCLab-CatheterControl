import torch
import camera_settings
import interspace_transforms as it


class Bezier():
    def __init__(self, p0, para, num_samples=200, r=0.01, downscale=1):
        '''
        find the 3d points on the bezier curve
        and the projected 2d points from the bezier curve

        Args:s
            p0: the start point of the bezier curve
            para: three control points of the bezier curve
            num_samples: the number of points on the bezier curve should be extracted
                        when calculate image loss, the recommanded value is 200
                        when calculate bezier shift loss, the recommanded value is 30

        Outputs:
            self.p3d_from_bezier: the 3d points on the bezier curve
            self.p2d_from_bezier: the 2d points projected from bezier curve
        '''
        self.p0 = p0
        self.para = para
        self.r = r

        self.num_samples = num_samples
        self.downscale = downscale

        # camera E parameters
        self.cam_RT_H = camera_settings.cam_RT_H

        # camera I parameters
        self.cam_K = camera_settings.cam_K
        self.cam_K = self.cam_K / self.downscale
        self.cam_K[-1, -1] = 1

        self.getBezier3pts()
        self.proj_to_2d()


    def getBezier3pts(self):
        '''
        using the three parameters to find the points on bezier curve

        Outputs:
            self.pos_bezier_3d 
        '''
        sample_list = torch.linspace(0,1,self.num_samples)

        P1 = self.p0
        PC = torch.hstack((self.para[0], self.para[1], self.para[2]))
        P2 = torch.hstack((self.para[3], self.para[4], self.para[5]))
        P1p = 2 / 3 * PC + 1 / 3 * P1
        P2p = 2 / 3 * PC + 1 / 3 * P2

        # Get positions from samples along bezier curve 
        pos_bezier_3d = torch.zeros(self.num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier_3d[i, :] = (1 - s)**3 * P1 + 3 * s * (1 - s)**2 * \
                P1p + 3 * (1 - s) * s**2 * P2p + s**3 * P2

        self.p3d_from_bezier = torch.flip(pos_bezier_3d, dims=[0]) 



    def proj_to_2d(self):
        '''
        project the found 3d points on bezier curve to 2d

        Outputs:
            self.pos_bezier_2d
        '''
        pos_bezier_3d = torch.flip(self.p3d_from_bezier, dims=[0])
        pos_bezier_3d_H = torch.cat((pos_bezier_3d, torch.ones(self.num_samples, 1)), dim=1)

        pos_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(pos_bezier_3d_H, 0, 1)), 0, 1)
        pos_bezier_cam = pos_bezier_cam_H[1:, :-1]

        if pos_bezier_cam.shape == (3, ):
            pos_bezier_cam = torch.unsqueeze(pos_bezier_cam, dim=0)

        divide_z = torch.div(torch.transpose(pos_bezier_cam[:, :-1], 0, 1), pos_bezier_cam[:, -1])

        divide_z = torch.cat((divide_z, torch.ones(1, pos_bezier_cam.shape[0])), dim=0)

        pos_bezier_2d = torch.transpose(torch.matmul(self.cam_K, divide_z)[:-1, :], 0, 1)

        self.p2d_from_bezier = torch.flip(pos_bezier_2d, dims=[0])

    
    def calculate_bezier_specs(self):
        '''
        when the points are changed
            e.g. after shift is applied
        the new bezier specs should be calculated
        '''
        p_mid = self.p3d_from_bezier[int(len(self.p3d_from_bezier)/2)]
        p_end = self.p3d_from_bezier[0]

        c = (p_mid - 0.25*(self.p0 + p_end))*2

        para_new = torch.zeros(6)
        para_new[:3] = c
        para_new[3:] = p_end

        self.para_new = para_new


    def apply_shift(self, shift):
        '''
        Apply shift to Bezier Curve
        To get points on new Bezier Curve

        Inputs:
            shift ((3,) tensor): dux, duy, dl
            num_samples: the points should be got from the Bezier Curve
        '''
        dux = torch.clone(shift[0]) 
        duy = torch.clone(shift[1]) 
        dl = torch.clone(shift[2]) 

        p3d_selected = torch.zeros(self.num_samples + 1, 3)
        for i in range(self.num_samples + 1):
            p3d_selected[i,:] = self.p3d_from_bezier[int(i*len(self.p3d_from_bezier)/(self.num_samples+1))]
        p3d_selected = p3d_selected[:-1]

        p_start = self.p0
        p3d_shifted_list = torch.zeros(self.num_samples, 3)

        for i, p3d in enumerate(p3d_selected):
            # weight = torch.tensor((self.num_samples-i)/self.num_samples)
            para_phithetal = it.get_phithetal_from_bezier(p_start, p3d)
            para_uxuyl = it.get_uxuyl_from_phithetal(para_phithetal[0], para_phithetal[1], para_phithetal[2], self.r)
            # ux_2, uy_2, l_2 = para_uxuyl[0]-dux*weight, para_uxuyl[1]-duy*weight, para_uxuyl[2]-dl*weight
            ux_2, uy_2, l_2 = para_uxuyl[0]-dux, para_uxuyl[1]-duy, para_uxuyl[2]-dl
            p3d_shifted_list[i, :] = it.get_point_from_uxuyl(p_start, ux_2, uy_2, l_2, self.r)

        self.p3d_from_bezier = p3d_shifted_list

        self.proj_to_2d()

        self.calculate_bezier_specs()







