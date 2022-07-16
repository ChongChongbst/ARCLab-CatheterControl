import torch

def inverse_kinematic_3dof(p_start, p_end):
    '''
    Calculate interspace transformation from cc curve end point to configuration [phi, theta, l]
                                    from end point to configuration space

    Args:
        p_end ((3,) tensor): the end point of the constant curvature curve

    Return:
        the parameters in configuration space
    '''
    x = p_end[0] - p_start[0]
    y = p_end[1] - p_start[1]
    z = p_end[2] - p_start[2]

    phi = torch.atan(y/x)

    k = 2*torch.sqrt(x*x+y*y)/(x*x+y*y+z*z)

    if z > 0:
        theta = torch.acos(1-k*torch.sqrt(x*x+y*y))
    else:
        theta = 2*torch.pi - torch.acos(1-k*torch.sqrt(x*x+y*y))

    l = theta/k

    return torch.tensor([phi, theta, l])

def para_transform_3dof(phi, theta, l, r):
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

    return torch.tensor([ux, uy, l])


def cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    '''
    Calculate constant curvature 3DoF transformation

    Args:
        p_0 ((,3) tensor): start point of catheter
        ux (float tensor): 1st pair of tendon length (responsible for catheter bending)
        uy (float tensor): 2nd pair of tendon length (responsible for catheter bending)
        l (float tensor): length of catheter
        r (float tensor): cross section radius of catheter
        s (float tensor from 0 to 1 inclusive): s value representing position on the cc curve
    '''
    p_0_4d = torch.cat((p_0, torch.tensor([1])), 0)
    u = torch.sqrt(ux*ux +uy*uy)
    theta = u / r * s
    l = l * s

    T = torch.tensor( [ [1+ux*ux/(u*u)*(torch.cos(theta)-1), ux*uy/(u*u)*(torch.cos(theta)-1),   -1*ux/u*torch.sin(theta), r*l*ux*(1-torch.cos(theta))/(u*u)],
                        [ux*uy/(u*u)*(torch.cos(theta)-1),   1+uy*uy/(u*u)*(torch.cos(theta)-1), -1*uy/u*torch.sin(theta), r*l*uy*(1-torch.cos(theta))/(u*u)],
                        [ux/u*torch.sin(theta),              uy/u*torch.sin(theta),              torch.cos(theta),         r*l*torch.sin(theta)/u           ],
                        [0,                                  0,                                  0,                        1                                ]])

    return torch.matmul(T, p_0_4d)[:3]