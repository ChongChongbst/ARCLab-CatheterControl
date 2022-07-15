import torch

def convert_bezier_to_cc(p_0, bezier_specs):
    '''
    Convert a Bezier curve back to a constant curvature curve

    Args:
        p_0 ((3), tensor): the start point of Bezier Curve
        bezier_specs ((6) tensor): the first 3 is composed of the middel control point of Bezier Curve
                            the last 3 is composed of the end control point of the Bezier Curve

    '''
    p_start = p_0
    c = torch.tensor([bezier_specs[0], bezier_specs[1], bezier_specs[2]])
    p_end = torch.tensor([bezier_specs[3], bezier_specs[4], bezier_specs[5]])
    p_mid = (c / 2) + (p_start / 4) + (p_end / 4)

    return p_end

def calculate_inverse_jacobian_2dof_theta_phi(p_start, p_end, l, ):
    '''
    Calculate Inverse Jacobian of 2DoF interspace control with (theta, phi) parameterization

    Args:
        p_start ((3,) tensor): start point of catheter
        p_end ((3,) tensor): end point of catheter

    Note:
        theta and phi specify the 2 dimensional catheter bending

    Returns:
        ((2, 6) tensor): Inverse Jacobian of 2DoF interspace control with  

    '''