import torch
from skeleton import Skeleton
from bezier import Bezier

class Img_bezier_loss():
    def __init__(self, skeleton, bezier, weight, num_samples=200, downscale=1):
        '''
        give a image and a bezier curve parameter
        return the 2d loss between them

        Args:
            skeleton: the extracted skeleton from image
            bezier: Bezier object from "bezier.py"
            weight((2,) tensor): the weight of centerline loss and tip loss

        Outputs:
            self.loss
        '''
        self.skeleton = skeleton

        self.p2d_from_bezier = bezier.p2d_from_bezier
        self.p3d_from_bezier = bezier.p3d_from_bezier

        self.weight = weight

        self.find_arranged_centerline()
        self.get_loss_centerline()



    def find_arranged_centerline(self):
        '''
        Use the centerline points got directly from the image
        to find the projected centerline of the 3D bezier curve fitted with the same size

        Variables:
            self.p2d_from_bezier: the 2d points projected from bezier curve
            self.skeleton: the centerline extracted from the image taken from monocular camera

        Result:
            self.centerline: the closest point on the projected fitted curve to the skeleton points on the image      
        '''
        skeleton = torch.clone(self.skeleton)
        p2d_from_bezier = torch.clone(self.p2d_from_bezier)

        centerline = []
        for i in range(skeleton.shape[0]):
            err = torch.linalg.norm(skeleton[i] - p2d_from_bezier, ord=None, axis=1)
            index = torch.argmin(err)
            temp = p2d_from_bezier[index, ]
            centerline.append(temp)

        self.centerline = torch.stack(centerline)


    def get_loss_centerline(self):
        '''
        find the 2d loss 
        '''
        skeleton = torch.clone(self.skeleton)
        p2d_from_bezier = torch.clone(self.p2d_from_bezier)

        err_skeleton_by_corresp = torch.linalg.norm(skeleton - self.centerline, ord=None, axis=1) / 1.0
        err_skeleton_sum_by_corresp = torch.sum(err_skeleton_by_corresp) / p2d_from_bezier.shape[0]

        err_obj_Tip = torch.linalg.norm(skeleton[0, :] - p2d_from_bezier[0, :], ord=None)

        self.loss = err_skeleton_sum_by_corresp*self.weight[0] + err_obj_Tip*self.weight[1]      
    
