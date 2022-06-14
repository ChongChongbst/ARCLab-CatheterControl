import numpy as np
import transforms



def bezier_from_cc(s_bezier, p_start, p_mid, p_end):
    """
    Get point on Bezier curve given the start point, middle control point, end point, and s value

    Args:
        s_bezier (float from 0 to 1 inclusive): s value
        p_start ((3,) numpy array): start point of Bezier curve
        p_mid ((3,) numpy array): middle control point of Bezier curve
        p_end ((3,) numpy array): end point of Bezier curve

    Returns:
        ((3,) numpy array) a point on the Bezier curve
    """
    return p_start * (s_bezier - 1) ** 2 + p_mid * 4 * s_bezier * (1 - s_bezier) + p_end * s_bezier * (3 * s_bezier - 2)


def d_bezier_d_p_start(s_bezier):
    """
    Calculate derivative of Bezier curve with respect to the start point

    Args:
        s_bezier (float from 0 to 1 inclusive): s value

    Returns:
        (float) derivative of Bezier curve with respect to the start point
    """
    return (s_bezier - 1) ** 2


def d_bezier_d_p_mid(s_bezier):
    """
    Calculate derivative of Bezier curve with respect to the middle control point

    Args:
        s_bezier (float from 0 to 1 inclusive): s value

    Returns:
        (float) derivative of Bezier curve with respect to the middle control point
    """
    return 4 * s_bezier * (1 - s_bezier)


def d_bezier_d_p_end(s_bezier):
    """
    Calculate derivative of Bezier curve with respect to the end point

    Args:
        s_bezier (float from 0 to 1 inclusive): s value

    Returns:
        (float) derivative of Bezier curve with respect to the end point
    """
    return s_bezier * (3 * s_bezier - 2)


def calculate_jacobian_2dof_ux_uy(p_start, ux, uy, l, r):
    """
    Calculate Jacobian of 2DoF interspace control with (ux, uy) parameterization

    Args:
        p_start ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter

    Returns:
        ((6, 2) numpy array): Jacobian of 2DoF interspace control with (ux, uy) parameterization
    """
    bezier_mid_over_cc_start = d_bezier_d_p_start(0.5)
    bezier_end_over_cc_start = d_bezier_d_p_start(1.0)
    bezier_mid_over_cc_mid = d_bezier_d_p_mid(0.5)
    bezier_end_over_cc_mid = d_bezier_d_p_mid(1.0)
    bezier_mid_over_cc_end = d_bezier_d_p_end(0.5)
    bezier_end_over_cc_end = d_bezier_d_p_end(1.0)

    cc_start_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_start_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_mid_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_mid_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_end_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)
    cc_end_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)


    G_p_mid_ux = bezier_mid_over_cc_start * cc_start_over_ux + bezier_mid_over_cc_mid * cc_mid_over_ux + bezier_mid_over_cc_end * cc_end_over_ux
    G_p_mid_uy = bezier_mid_over_cc_start * cc_start_over_uy + bezier_mid_over_cc_mid * cc_mid_over_uy + bezier_mid_over_cc_end * cc_end_over_uy
    G_p_end_ux = bezier_end_over_cc_start * cc_start_over_ux + bezier_end_over_cc_mid * cc_mid_over_ux + bezier_end_over_cc_end * cc_end_over_ux
    G_p_end_uy = bezier_end_over_cc_start * cc_start_over_uy + bezier_end_over_cc_mid * cc_mid_over_uy + bezier_end_over_cc_end * cc_end_over_uy

    J = np.zeros((6, 2))
    J[0:3, 0] = G_p_mid_ux
    J[0:3, 1] = G_p_mid_uy
    J[3:6, 0] = G_p_end_ux
    J[3:6, 1] = G_p_end_uy

    return J


def calculate_jacobian_3dof_ux_uy(p_start, ux, uy, l, r):
    """
    Calculate Jacobian of 3DoF interspace control with (ux, uy) parameterization

    Args:
        p_start ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter

    Returns:
        ((6, 3) numpy array): Jacobian of 3DoF interspace control with (ux, uy) parameterization
    """
    bezier_mid_over_cc_start = d_bezier_d_p_start(0.5)
    bezier_end_over_cc_start = d_bezier_d_p_start(1.0)
    bezier_mid_over_cc_mid = d_bezier_d_p_mid(0.5)
    bezier_end_over_cc_mid = d_bezier_d_p_mid(1.0)
    bezier_mid_over_cc_end = d_bezier_d_p_end(0.5)
    bezier_end_over_cc_end = d_bezier_d_p_end(1.0)

    cc_start_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_start_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_start_over_l = transforms.d_l_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_mid_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_mid_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_mid_over_l = transforms.d_l_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_end_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)
    cc_end_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)
    cc_end_over_l = transforms.d_l_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)

    G_p_mid_ux = bezier_mid_over_cc_start * cc_start_over_ux + bezier_mid_over_cc_mid * cc_mid_over_ux + bezier_mid_over_cc_end * cc_end_over_ux
    G_p_mid_uy = bezier_mid_over_cc_start * cc_start_over_uy + bezier_mid_over_cc_mid * cc_mid_over_uy + bezier_mid_over_cc_end * cc_end_over_uy
    G_p_mid_l = bezier_mid_over_cc_start * cc_start_over_l + bezier_mid_over_cc_mid * cc_mid_over_l + bezier_mid_over_cc_end * cc_end_over_l
    G_p_end_ux = bezier_end_over_cc_start * cc_start_over_ux + bezier_end_over_cc_mid * cc_mid_over_ux + bezier_end_over_cc_end * cc_end_over_ux
    G_p_end_uy = bezier_end_over_cc_start * cc_start_over_uy + bezier_end_over_cc_mid * cc_mid_over_uy + bezier_end_over_cc_end * cc_end_over_uy
    G_p_end_l = bezier_end_over_cc_start * cc_start_over_l + bezier_end_over_cc_mid * cc_mid_over_l + bezier_end_over_cc_end * cc_end_over_l

    J = np.zeros((6, 3))
    J[0:3, 0] = G_p_mid_ux
    J[0:3, 1] = G_p_mid_uy
    J[0:3, 2] = G_p_mid_l
    J[3:6, 0] = G_p_end_ux
    J[3:6, 1] = G_p_end_uy
    J[3:6, 2] = G_p_end_l

    return J


def calculate_ux(theta, phi, r):
    """
    Calculate ux from theta, phi, and r

    Args:
        theta (float): (in radians) theta parameter
        phi (float): (in radians) phi parameter
        r (float): cross section radius of catheter
    
    Note:
        theta and phi together specify the 2 dimensional catheter bending

    Returns:
        ux (float): 1st pair of tendon length (responsible for catheter bending)
    """
    return theta * r * np.cos(phi)


def calculate_uy(theta, phi, r):
    """
    Calculate uy from theta, phi, and r

    Args:
        theta (float): (in radians) theta parameter
        phi (float): (in radians) phi parameter
        r (float): cross section radius of catheter
    
    Note:
        theta and phi together specify the 2 dimensional catheter bending

    Returns:
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
    """
    return theta * r * np.sin(phi)


def d_ux_d_theta(phi, r):
    """
    Calculate derivative of ux with respect to theta
    """
    return r * np.cos(phi)


def d_uy_d_theta(phi, r):
    """
    Calculate derivative of uy with respect to theta
    """
    return r * np.sin(phi)


def d_ux_d_phi(theta, phi, r):
    """
    Calculate derivative of ux with respect to phi
    """
    return -1 * theta * r * np.sin(phi)


def d_uy_d_phi(theta, phi, r):
    """
    Calculate derivative of uy with respect to phi
    """
    return theta * r * np.cos(phi)


def calculate_jacobian_2dof_theta_phi(p_start, theta, phi, l, r):
    """
    Calculate Jacobian of 2DoF interspace control with (theta, phi) parameterization

    Args:
        p_start ((3,) numpy array): start point of catheter
        theta (float): (in radians) theta parameter
        phi (float): (in radians) phi parameter
        l (float): length of catheter
        r (float): cross section radius of catheter

    Note:
        theta and phi together specify the 2 dimensional catheter bending

    Returns:
        ((6, 2) numpy array): Jacobian of 2DoF interspace control with (theta, phi) parameterization
    """
    ux = calculate_ux(theta, phi, r)
    uy = calculate_uy(theta, phi, r)

    ux_over_phi = d_ux_d_phi(theta, phi, r)
    uy_over_phi = d_uy_d_phi(theta, phi, r)
    ux_over_theta = d_ux_d_theta(phi, r)
    uy_over_theta = d_uy_d_theta(phi, r)
    
    bezier_mid_over_cc_start = d_bezier_d_p_start(0.5)
    bezier_end_over_cc_start = d_bezier_d_p_start(1.0)
    bezier_mid_over_cc_mid = d_bezier_d_p_mid(0.5)
    bezier_end_over_cc_mid = d_bezier_d_p_mid(1.0)
    bezier_mid_over_cc_end = d_bezier_d_p_end(0.5)
    bezier_end_over_cc_end = d_bezier_d_p_end(1.0)

    cc_start_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_start_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_mid_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_mid_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_end_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)
    cc_end_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)

    cc_start_over_theta = cc_start_over_ux * ux_over_theta + cc_start_over_uy * uy_over_theta
    cc_mid_over_theta = cc_mid_over_ux * ux_over_theta + cc_mid_over_uy * uy_over_theta
    cc_end_over_theta = cc_end_over_ux * ux_over_theta + cc_end_over_uy * uy_over_theta
    cc_start_over_phi = cc_start_over_ux * ux_over_phi + cc_start_over_uy * uy_over_phi
    cc_mid_over_phi = cc_mid_over_ux * ux_over_phi + cc_mid_over_uy * uy_over_phi
    cc_end_over_phi = cc_end_over_ux * ux_over_phi + cc_end_over_uy * uy_over_phi

    G_p_mid_theta = bezier_mid_over_cc_start * cc_start_over_theta + bezier_mid_over_cc_mid * cc_mid_over_theta + bezier_mid_over_cc_end * cc_end_over_theta
    G_p_mid_phi = bezier_mid_over_cc_start * cc_start_over_phi + bezier_mid_over_cc_mid * cc_mid_over_phi + bezier_mid_over_cc_end * cc_end_over_phi
    G_p_end_theta = bezier_end_over_cc_start * cc_start_over_theta + bezier_end_over_cc_mid * cc_mid_over_theta + bezier_end_over_cc_end * cc_end_over_theta
    G_p_end_phi = bezier_end_over_cc_start * cc_start_over_phi + bezier_end_over_cc_mid * cc_mid_over_phi + bezier_end_over_cc_end * cc_end_over_phi

    J = np.zeros((6, 2))
    J[0:3, 0] = G_p_mid_theta
    J[0:3, 1] = G_p_mid_phi
    J[3:6, 0] = G_p_end_theta
    J[3:6, 1] = G_p_end_phi

    return J


def calculate_jacobian_3dof_theta_phi(p_start, theta, phi, l, r):
    """
    Calculate Jacobian of 3DoF interspace control with (theta, phi) parameterization

    Args:
        p_start ((3,) numpy array): start point of catheter
        theta (float): (in radians) theta parameter
        phi (float): (in radians) phi parameter
        l (float): length of catheter
        r (float): cross section radius of catheter

    Note:
        theta and phi together specify the 2 dimensional catheter bending

    Returns:
        ((6, 3) numpy array): Jacobian of 3DoF interspace control with (theta, phi) parameterization
    """
    ux = calculate_ux(theta, phi, r)
    uy = calculate_uy(theta, phi, r)

    ux_over_phi = d_ux_d_phi(theta, phi, r)
    uy_over_phi = d_uy_d_phi(theta, phi, r)
    ux_over_theta = d_ux_d_theta(phi, r)
    uy_over_theta = d_uy_d_theta(phi, r)
    
    bezier_mid_over_cc_start = d_bezier_d_p_start(0.5)
    bezier_end_over_cc_start = d_bezier_d_p_start(1.0)
    bezier_mid_over_cc_mid = d_bezier_d_p_mid(0.5)
    bezier_end_over_cc_mid = d_bezier_d_p_mid(1.0)
    bezier_mid_over_cc_end = d_bezier_d_p_end(0.5)
    bezier_end_over_cc_end = d_bezier_d_p_end(1.0)

    cc_start_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_start_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_start_over_l = transforms.d_l_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_mid_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_mid_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_mid_over_l = transforms.d_l_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_end_over_ux = transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)
    cc_end_over_uy = transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)
    cc_end_over_l = transforms.d_l_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)

    cc_start_over_theta = cc_start_over_ux * ux_over_theta + cc_start_over_uy * uy_over_theta
    cc_mid_over_theta = cc_mid_over_ux * ux_over_theta + cc_mid_over_uy * uy_over_theta
    cc_end_over_theta = cc_end_over_ux * ux_over_theta + cc_end_over_uy * uy_over_theta
    cc_start_over_phi = cc_start_over_ux * ux_over_phi + cc_start_over_uy * uy_over_phi
    cc_mid_over_phi = cc_mid_over_ux * ux_over_phi + cc_mid_over_uy * uy_over_phi
    cc_end_over_phi = cc_end_over_ux * ux_over_phi + cc_end_over_uy * uy_over_phi

    G_p_mid_theta = bezier_mid_over_cc_start * cc_start_over_theta + bezier_mid_over_cc_mid * cc_mid_over_theta + bezier_mid_over_cc_end * cc_end_over_theta
    G_p_mid_phi = bezier_mid_over_cc_start * cc_start_over_phi + bezier_mid_over_cc_mid * cc_mid_over_phi + bezier_mid_over_cc_end * cc_end_over_phi
    G_p_mid_l = bezier_mid_over_cc_start * cc_start_over_l + bezier_mid_over_cc_mid * cc_mid_over_l + bezier_mid_over_cc_end * cc_end_over_l
    G_p_end_theta = bezier_end_over_cc_start * cc_start_over_theta + bezier_end_over_cc_mid * cc_mid_over_theta + bezier_end_over_cc_end * cc_end_over_theta
    G_p_end_phi = bezier_end_over_cc_start * cc_start_over_phi + bezier_end_over_cc_mid * cc_mid_over_phi + bezier_end_over_cc_end * cc_end_over_phi
    G_p_end_l = bezier_end_over_cc_start * cc_start_over_l + bezier_end_over_cc_mid * cc_mid_over_l + bezier_end_over_cc_end * cc_end_over_l

    J = np.zeros((6, 3))
    J[0:3, 0] = G_p_mid_theta
    J[0:3, 1] = G_p_mid_phi
    J[0:3, 2] = G_p_mid_l
    J[3:6, 0] = G_p_end_theta
    J[3:6, 1] = G_p_end_phi
    J[3:6, 2] = G_p_end_l

    return J
