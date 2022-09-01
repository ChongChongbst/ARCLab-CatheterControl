import torch
torch.set_printoptions(precision=8)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_phithetal_from_bezier(p_start, p_end):
    '''
    Calculate interspace transformation from cc curve end point to configuration [phi, theta, l]
                                    from end point to configuration space

    Args:
        p_end ((3,) tensor): the end point of the constant curvature curve

    Return:
        the parameters in configuration space
    '''
    x = p_end[0]-p_start[0]
    y = p_end[1]-p_start[1]
    z = p_end[2]-p_start[2]

    if x < 0:
        phi = torch.pi + torch.atan(y/x)
    else:
        phi = torch.atan(y/x)

    r = (x*x+y*y+z*z)/(2*torch.sqrt(x*x+y*y))

    if z > 0:
        theta = torch.acos(1-torch.sqrt(x*x+y*y)/r)
    else:
        theta = 2*torch.pi-torch.acos(1-torch.sqrt(x*x+y*y)/r)

    l = theta*r

    return torch.hstack((phi, theta, l))

def get_uxuyl_from_phithetal(phi, theta, l, r):
    '''
    Calculate interspace transformation from [phi, theta, l] configuration to [ux, uy, l] configuration
                        from configuration space to actuator space

    Args:
        phi (float tensor): the angle of rotation moves the robot out the x-z plane
        theta (float tensor): the angle of rotaion about the y-axis
        l (float tensor): length of catheter
        r (float tensor): cross section radius of catheter 
    '''

    ux = theta * r * torch.cos(phi)
    uy = theta * r * torch.sin(phi)

    return torch.hstack((ux, uy, l))

def para1_transform_3dof(p_start, phi, theta, l, s=1):
    '''
    Calculate constant curvature 3DoF transformation
    '''
    p_0_4d = torch.cat((torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1])), 0)
    k=theta/l
    theta = theta * s

    T = torch.tensor([[1+torch.cos(phi)*torch.cos(phi)*(torch.cos(theta)-1),  torch.sin(phi)*torch.cos(phi)*(torch.cos(theta)-1),                  torch.cos(phi)*torch.sin(theta), torch.cos(phi)*(1-torch.cos(theta))/k],
                      [torch.sin(phi)*torch.cos(phi)*(torch.cos(theta)-1),    torch.cos(phi)*torch.cos(phi)*(1-torch.cos(theta))+torch.cos(theta), torch.sin(phi)*torch.sin(theta), torch.sin(phi)*(1-torch.cos(theta))/k],
                      [-torch.cos(phi)*torch.sin(theta),                      -torch.sin(phi)*torch.sin(theta),                                    torch.cos(theta),                torch.sin(theta)/k                   ],
                      [0,                                                     0,                                                                   0,                               1 ]])

    p3d = torch.matmul(T, p_0_4d)[:3]+p_start

    return p3d



def get_point_from_uxuyl(p_start, ux, uy, l, r, s=1):
    '''
    Calculate constant curvature 3DoF transformation

    Args:
        p_start ((,3) tensor): start point of catheter
        ux (float tensor): 1st pair of tendon length (responsible for catheter bending)
        uy (float tensor): 2nd pair of tendon length (responsible for catheter bending)
        l (float tensor): length of catheter
        r (float tensor): cross section radius of catheter
        s (float tensor from 0 to 1 inclusive): s value representing position on the cc curve
    '''
    #p_0_4d = torch.cat((torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1])), 0)
    coordinate_0_1 = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    coordinate_0_2 = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    coordinate_0_3 = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    coordinate_0_4 = torch.vstack((p_start[0], p_start[1], p_start[2], torch.tensor([1])))
    coordinate_0 = torch.hstack((coordinate_0_1, coordinate_0_2, coordinate_0_3, coordinate_0_4))


    #p_0_4d = torch.cat((p_start, torch.tensor([1])), 0)
    u = torch.sqrt(ux*ux +uy*uy)
    theta = u / r * s
    l = l

    T_1 = torch.hstack((1+ux*ux/(u*u)*(torch.cos(theta)-1), ux*uy/(u*u)*(torch.cos(theta)-1),   ux/u*torch.sin(theta), r*l*ux*(1-torch.cos(theta))/(u*u)))
    T_2 = torch.hstack((ux*uy/(u*u)*(torch.cos(theta)-1),   1+uy*uy/(u*u)*(torch.cos(theta)-1), uy/u*torch.sin(theta), r*l*uy*(1-torch.cos(theta))/(u*u)))
    T_3 = torch.hstack((-ux/u*torch.sin(theta),             -uy/u*torch.sin(theta),             torch.cos(theta),      r*l*torch.sin(theta)/u           ))
    T_4 = torch.hstack((-torch.tensor(0),                   -torch.tensor(0),                   torch.tensor(0),       torch.tensor(1)                  ))
    
    T = torch.vstack((T_1, T_2, T_3, T_4))

    #p3d = p_start+torch.matmul(T, p_0_4d)[:3]
    #p3d = torch.matmul(T, p_0_4d)[:3]

    coordinate_1 = torch.matmul(coordinate_0,T)
    p3d = coordinate_1[:3,3]

    return p3d


def get_pts_from_uxuyl(p_start, ux, uy, l, r, num_samples=100):
    '''
    find a list of point with the same interval according to configuration ux, uy and l

    Args:
        p_start ((,3) tensor): start point of catheter
        ux (float tensor): 1st pair of tendon length (responsible for catheter bending)
        uy (float tensor): 2nd pair of tendon length (responsible for catheter bending)
        l (float tensor): length of catheter
        r (float tensor): cross section radius of catheter
        num_samples (int): the number of points on the curve 
    '''
    sample_list = torch.flip(torch.linspace(0, 1, num_samples+1), dims=[0])[:-1]

    p3d_list = torch.zeros(num_samples, 3)
    for i, s in enumerate(sample_list):
        p3d_list[i, :] = get_point_from_uxuyl(p_start, ux, uy, l, r, s)

    return p3d_list

def multiple_shift(p_start, ux, uy, l, r, shift_list=torch.tensor([0,1e-4, 5e-4]), s=1):
    '''
    find multiple frames with different shift

    Args:
        p_start ((,3) tensor): start point of catheter
        ux (float tensor): 1st pair of tendon length (responsible for catheter bending)
        uy (float tensor): 2nd pair of tendon length (responsible for catheter bending)
        l (float tensor): length of catheter
        r (float tensor): cross section radius of catheter
    '''
    shift_list = torch.clone(shift_list)
    num_shift = shift_list.size(0) * shift_list.size(0) - 1
    p_mul_shifted = torch.zeros(num_shift, 3)

    t = 0
    for i in range(shift_list.size(0)):
        for j in range(shift_list.size(0)):
            if i==0 and j==0:
                continue
            p_mul_shifted[t,:]=get_point_from_uxuyl(p_start, ux+shift_list[i], uy+shift_list[j], l, r, s)
            t += 1

    return p_mul_shifted


def para2_mul_shifted_3dof_list(p_start, ux, uy, l, r, shift_list=torch.tensor([0,1e-4, 5e-4]), num_samples=100):
    '''
    find a list of point with the same interval and shifted with multiple value 
    according to configuration ux, uy and l

    Args:
        p_start ((,3) tensor): start point of catheter
        ux (float tensor): 1st pair of tendon length (responsible for catheter bending)
        uy (float tensor): 2nd pair of tendon length (responsible for catheter bending)
        l (float tensor): length of catheter
        r (float tensor): cross section radius of catheter
        num_samples (int): the number of points on the curve 
    '''
    shift_list = torch.clone(shift_list)
    num_shift = shift_list.size(0)*shift_list.size(0)-1
    sample_list = torch.flip(torch.linspace(0, 1, num_samples+1), dims=[0])[:-1]

    p3d_list = torch.zeros(num_shift, num_samples, 3)
    for i, s in enumerate(sample_list):
        p3d_list[:, i, :] = multiple_shift(p_start, ux, uy, l, r, shift_list, s)

    return p3d_list

    


if __name__ == '__main__':
    '''
    main debuging program: to test if all the function aboved is calculated correctly
    '''
    p_start = torch.tensor([0.02, 0.002, 0.00000000])
    p_end = torch.tensor([-0.02258245, -0.04751587,  0.35071838], requires_grad=True)
    r = 0.01

    [phi, theta, l] = get_phithetal_from_bezier(p_start, p_end)

    print('l:'+str(l))
    [ux, uy, l] = get_uxuyl_from_phithetal(phi, theta, r)
    p_end_1 = para1_transform_3dof(p_start, phi, theta, l)
    print('ux:'+str(ux))
    print('uy:'+str(uy))
    p_end_2 = get_point_from_uxuyl(p_start, ux, uy, l, r)
    

    loss = torch.linalg.norm(p_end-p_end_2)

    config=torch.tensor([-3.00644210e-03, -1.41444699e-03,  3.92639322e-01])
    p_end_3 = get_point_from_uxuyl(p_start, config[0], config[1], config[2], r)

    
    print(loss)
    print(p_end_1)
    print(p_end_2)
    print(p_end_3)

    print('--------------test2-----------------start---------------')
    p3d_list = get_pts_from_uxuyl(p_start, config[0], config[1], config[2], r, 25)
    p3d_list_np = p3d_list.detach().numpy()

    p3d_shift_list = get_pts_from_uxuyl(p_start, config[0]+1e-2, config[1]+1e-2, config[2], r, 25)
    p3d_shift_list_np = p3d_shift_list.detach().numpy()

    x = p3d_list_np[:, 0]
    y = p3d_list_np[:, 1]
    z = p3d_list_np[:, 2]

    x_shift = p3d_shift_list_np[:, 0]
    y_shift = p3d_shift_list_np[:, 1]
    z_shift = p3d_shift_list_np[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x,y,z, c='r')
    ax.scatter(x_shift,y_shift,z_shift, c='g')

    ax.set_xlabel('X', fontdict={'size':15, 'color':'red'})
    ax.set_ylabel('Y', fontdict={'size':15, 'color':'red'})
    ax.set_zlabel('Z', fontdict={'size':15, 'color':'red'})

    scale = matplotlib.ticker.MultipleLocator(0.01)
    ax = plt.gca()
    ax.xaxis.set_major_locator(scale)
    ax.yaxis.set_major_locator(scale)
    ax.zaxis.set_major_locator(scale)

    for i in range(len(p3d_list_np)):
        ax.plot([x[i],x_shift[i]], [y[i],y_shift[i]], [z[i],z_shift[i]], c='b')

    plt.gca().set_box_aspect((2,1,10))
    plt.show()