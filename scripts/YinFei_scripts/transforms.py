import numpy as np



def world_to_image_transform(p, camera_extrinsics, fx, fy, cx, cy):
    """
    Convert 3D point to 2D given camera parameters

    Args:
        p ((3,) numpy array): a point in 3D
        camera_extrinsics ((4, 4) numpy array): RT matrix 
        fx (float): horizontal direction focal length
        fy (float): vertical direction focal length
        cx (float): horizontal center of image
        cy (float): vertical center of image
    """
    p_4d = np.append(p, 1)
    p_cam = camera_extrinsics @ p_4d

    p_x = p_cam[0] * fx / p_cam[2] + cx
    p_y = p_cam[1] * fy / p_cam[2] + cy

    return np.array([p_x, p_y])


def image_to_world_transform(p_2d, camera_extrinsics, fx, fy, cx, cy, z):
    """
    Convert 2D point to 3D given camera parameters and depth

    Args:
        p_2d ((2,) numpy array): a point in 2D
        camera_extrinsics ((4, 4) numpy array): RT matrix
        fx (float): horizontal direction focal length
        fy (float): vertical direction focal length
        cx (float): horizontal center of image
        cy (float): vertical center of image
        z (float): depth
    """
    p_cam_x = (p_2d[0] - cx) * z / fx
    p_cam_y = (p_2d[1] - cy) * z / fy
    p_cam = np.array([p_cam_x, p_cam_y, z, 1])

    p_3d = (np.linalg.inv(camera_extrinsics) @ p_cam)[:-1]

    return p_3d


def world_to_image_interaction_matrix(p, camera_extrinsics, fx, fy):
    """
    Calculate world to image interaction matrix. This is used for inverse Jacobian with 2D loss

    Args:
        p ((3,) numpy array): a point in 3D
        camera_extrinsics ((4, 4) numpy array): RT matrix 
        fx (float): horizontal direction focal length
        fy (float): vertical direction focal length
    """
    p_4d = np.append(p, 1)
    p_cam = camera_extrinsics @ p_4d

    p_X = p_cam[0]
    p_Y = p_cam[1]
    p_Z = p_cam[2]

    #L = np.array([
    #    [-1 * fx / p_Z, 0, fx * p_X / p_Z / p_Z],
    #    [0, -1 * fy / p_Z, fy * p_Y / p_Z / p_Z]])

    L = np.array([
        [-1 * fx / p_Z, 0, -1 * fx * p_X / p_Z / p_Z],
        [0, -1 * fy / p_Z, -1 * fy * p_Y / p_Z / p_Z]])
    
    return L


def cc_transform_1dof(p_0, phi, u, l, r, s=1):
    """
    Calculate constant curvature 1DoF transformation
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        phi (float): phi (float): (in radians) phi parameter
        u (float): tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    k = u / r

    T = np.array([
        [np.cos(phi) ** 2 * (np.cos(k * s) - 1) + 1,      np.sin(phi) * np.cos(phi) * (np.cos(k * s) - 1),        np.cos(phi) * np.sin(k * s), np.cos(phi) * (1 - np.cos(k * s)) / k],
        [np.sin(phi) * np.cos(phi) * (np.cos(k * s) - 1), np.cos(phi) ** 2 * (1 - np.cos(k * s)) + np.cos(k * s), np.sin(phi) * np.sin(k * s), np.sin(phi) * (1 - np.cos(k * s)) / k],
        [-1 * np.cos(phi) * np.sin(k * s),                -1 * np.sin(phi) * np.sin(k * s),                       np.cos(k * s),               np.sin(k * s) / k                    ],
        [0,                                               0,                                                      0,                           1                                    ]])

    return (T @ p_0_4d)[:3]


def cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    """
    Calculate constant curvature 3DoF transformation
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    u = np.sqrt(ux ** 2 + uy ** 2)
    k = u / r

    T = np.array([
        [1 + (ux ** 2) / (u ** 2) * (np.cos(k * s) - 1), ux * uy / (u ** 2) * (np.cos(k * s) - 1),       -1 * ux / u * np.sin(k * s), r * l * ux * (1 - np.cos(k * s)) / (u ** 2)],
        [ux * uy / (u ** 2) * (np.cos(k * s) - 1),       1 + (uy ** 2) / (u ** 2) * (np.cos(k * s) - 1), -1 * uy / u * np.sin(k * s), r * l * uy * (1 - np.cos(k * s)) / (u ** 2)],
        [ux / u * np.sin(k * s),                         uy / u * np.sin(k * s),                         np.cos(k * s),               r * l * np.sin(k * s) / u                  ],
        [0,                                              0,                                              0,                           1                                          ]], dtype=object)

    return (T @ p_0_4d)[:3]


def d_u_cc_transform_1dof(p_0, phi, u, l, r, s=1):
    """
    Calculate derivative of constant curvature 1DoF transformation with respect to u
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        phi (float): phi (float): (in radians) phi parameter
        u (float): tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    a = s / l / r

    dT_du = np.array([
        [-1 * a * (np.cos(phi) ** 2) * np.sin(a * u),
         -1 * a * np.sin(phi) * np.cos(phi) * np.sin(a * u),
         a * np.cos(phi) * np.cos(a * u),
         -1 * l * r * np.cos(phi) / u / u + l * r * np.cos(phi) / u / u * np.cos(a * u) + l * r * np.cos(phi) / u * np.sin(a * u)],
        [-1 * a * np.sin(phi) * np.cos(phi) * np.sin(a * u),
         a * (np.cos(phi) ** 2) * np.sin(a * u) - a * np.sin(a * u),
         a * np.sin(phi) * np.cos(a * u),
         -1 * l * r * np.sin(phi) / u / u + l * r * np.sin(phi) / u / u * np.cos(a * u) + l * r * np.np.sin(phi) / u * np.sin(a * u)],
        [-1 * a * np.cos(phi) * np.cos(a * u),
         -1 * a * np.sin(phi) * np.cos(a * u),
         -1 * a * np.sin(a * u),
         -1 * l * r / u / u * np.sin(a * u) + l * r / u * np.cos(a * u)],
        [0, 0, 0, 0]])

    return (dT_du @ p_0_4d)[:3]


def d_ux_cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    """
    Calculate derivative of constant curvature 3DoF transformation with respect to ux
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    u = np.sqrt(ux ** 2 + uy ** 2)
    
    dT_dux = np.array([
        [(-2 * ux**3 * (-1 + np.cos((s * u) / r))) / (u**4) + (2 * ux * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux**3 * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * ux**2 * uy * (-1 + np.cos((s * u) / r))) / (u**4) + (uy * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux**2 * uy * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * ux**2 * np.cos((s * u) / r)) / (r * (u**2))) + (ux**2 * np.sin((s * u) / r)) / (u**3) - np.sin((s * u) / r) / u,
         (-2 * l * r * ux**2 * (1 - np.cos((s * u) / r))) / (u**4) + (l * r * (1 - np.cos((s * u) / r))) / (u**2) + (l * s * ux**2 * np.sin((s * u) / r)) / (u**3)],
        [(-2 * ux**2 * uy * (-1 + np.cos((s * u) / r))) / (u**4) + (uy * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux**2 * uy * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * ux * uy**2 * (-1 + np.cos((s * u) / r))) / (u**4) - (s * ux * uy**2 * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * ux * uy * np.cos((s * u) / r)) / (r * (u**2))) + (ux * uy * np.sin((s * u) / r)) / (u**3),
         (-2 * l * r * ux * uy * (1 - np.cos((s * u) / r))) / (u**4) + (l * s * ux * uy * np.sin((s * u) / r)) / (u**3)],
        [(s * ux**2 * np.cos((s * u) / r)) / (r * (u**2)) - (ux**2 * np.sin((s * u) / r)) / (u**3) + np.sin((s * u) / r) / u,
         (s * ux * uy * np.cos((s * u) / r)) / (r * (u**2)) - (ux * uy * np.sin((s * u) / r)) / (u**3),
         -((s * ux * np.sin((s * u) / r)) / (r * u)),
         (l * s * ux * np.cos((s * u) / r)) / (u**2) - (l * r * ux * np.sin((s * u) / r)) / (u**3)],
        [0, 0, 0, 0]])
        
    return (dT_dux @ p_0_4d)[:3]


def d_uy_cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    """
    Calculate derivative of constant curvature 3DoF transformation with respect to uy
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    u = np.sqrt(ux ** 2 + uy ** 2)
    
    dT_duy = np.array([
        [(-2 * ux**2 * uy * (-1 + np.cos((s * u) / r))) / (u**4) - (s * ux**2 * uy * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * ux * uy**2 * (-1 + np.cos((s * u) / r))) / (u**4) + (ux * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux * uy**2 * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * ux * uy * np.cos((s * u) / r)) / (r * (u**2))) + (ux * uy * np.sin((s * u) / r)) / (u**3),
         (-2 * l * r * ux * uy * (1 - np.cos((s * u) / r))) / (u**4) + (l * s * ux * uy * np.sin((s * u) / r)) / (u**3)],
        [(-2 * ux * uy**2 * (-1 + np.cos((s * u) / r))) / (u**4) + (ux * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux * uy**2 * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * uy**3 * (-1 + np.cos((s * u) / r))) / (u**4) + (2 * uy * (-1 + np.cos((s * u) / r))) / (u**2) - (s * uy**3 * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * uy**2 * np.cos((s * u) / r)) / (r * (u**2))) + (uy**2 * np.sin((s * u) / r)) / (u**3) - np.sin((s * u) / r) / u,
         (-2 * l * r * uy**2 * (1 - np.cos((s * u) / r))) / (u**4) + (l * r * (1 - np.cos((s * u) / r))) / (u**2) + (l * s * uy**2 * np.sin((s * u) / r)) / (u**3)],
        [(s * ux * uy * np.cos((s * u) / r)) / (r * (u**2)) - (ux * uy * np.sin((s * u) / r)) / (u**3),
         (s * uy**2 * np.cos((s * u) / r)) / (r * (u**2)) - (uy**2 * np.sin((s * u) / r)) / (u**3) + np.sin((s * u) / r) / u,
         -((s * uy * np.sin((s * u) / r)) / (r * u)),
         (l * s * uy * np.cos((s * u) / r)) / (u**2) - (l * r * uy * np.sin((s * u) / r)) / (u**3)],
        [0, 0, 0, 0]])
    
    return (dT_duy @ p_0_4d)[:3]


def d_l_cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    """
    Calculate derivative of constant curvature 3DoF transformation with respect to catheter length
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    u = np.sqrt(ux ** 2 + uy ** 2)

    dT_dl = np.array([
        [0, 0, 0, (r * ux * (1 - np.cos((s * u) / r))) / (u**2)],
        [0, 0, 0, (r * uy * (1 - np.cos((s * u) / r))) / (u**2)],
        [0, 0, 0, (r * np.sin((s * u) / r)) / u],
        [0, 0, 0, 0]])

    return (dT_dl @ p_0_4d)[:3]
