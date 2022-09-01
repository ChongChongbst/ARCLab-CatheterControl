import torch
import numpy as np
import os

from image_bezier_loss import Img_bezier_loss
from bezier import Bezier
from skeleton import Skeleton


def get_duxduydl(test_path, n_1, n_2):
    '''
    Find the dux, duy, dl
    '''
    para = np.load(test_path+'/params.npy')

    dux = para[n_2, 0] - para[n_1, 0]
    duy = para[n_2, 1] - para[n_1, 1]
    dl = para[n_2, 2] - para[n_1, 2]

    return torch.tensor([dux, duy, dl], dtype=torch.float)
    


def loss_3d_1(p0, para, test_path, weight, n=20):
    '''
    the first loss function
    '''
    num_samples = 30
    img_dir_path = test_path + "/images"

    loss = 0
    for i in range(n):
        shift = get_duxduydl(test_path, i, n)
        bzr = Bezier(p0, para, num_samples)
        bzr.apply_shift(shift)

        skeleton = Skeleton(img_dir_path, i).skeleton

        loss += torch.exp(torch.tensor(i))*Img_bezier_loss(skeleton, bzr, weight).loss / torch.tensor(n)

    return loss


def loss_3d_2(p0, para_list, test_path, n):
    '''
    the second loss function 
    '''
    num_samples = 41
    loss = 0
    for i in range(n):
        bzr_init = Bezier(p0, para_list[i], num_samples)
        bzr = Bezier(p0, para_list[i+1], num_samples)
        shift = get_duxduydl(test_path, i, i+1)
        bzr.apply_shift(shift)
        loss += torch.sum(torch.linalg.norm(bzr.p3d_from_bezier - bzr_init.p3d_from_bezier)) * torch.exp(torch.tensor(-(i+1)))
        loss += torch.sum(torch.linalg.norm(bzr.p3d_from_bezier[0] - bzr_init.p3d_from_bezier[0]))*torch.tensor(100)
        if i != 0:
            para_list[i] = bzr.para_new

    return loss

    




        



