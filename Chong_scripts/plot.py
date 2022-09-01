import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import os

import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

def find_image(test_path, i, downscale):
    '''
    test_path: the path to store all the data used for reconstruction
    i: the order of image used 
    '''
    img_dir_path = test_path + '/images'
    img_path = os.path.join(img_dir_path, str(i).zfill(3) + '.png')

    raw_img_rgb = cv2.imread(img_path)

    raw_img_rgb = cv2.resize(raw_img_rgb, (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))

    return raw_img_rgb



def plot_final_result(test_path, i, downscale, pos_bezier_3D, pos_bezier_3D_gt, pos_bezier_3D_init, error, proj_bezier_img, saved_opt_history):
    '''
    To plot the final result
    '''
    raw_img_rgb = find_image(test_path, i, downscale)
    
    centerline_draw_img_rgb = raw_img_rgb.copy()
    curve_3D_opt = pos_bezier_3D.detach().numpy()
    curve_3D_gt = pos_bezier_3D_gt.detach().numpy()
    curve_3D_init = pos_bezier_3D_init.detach().numpy()
    error_list = error.detach().numpy()
    error = np.linalg.norm(error_list)

    # Draw Centerline
    for i in range(proj_bezier_img.shape[0] - 1):
        p1 = (int(proj_bezier_img[i, 0]), int(proj_bezier_img[i, 1]))
        p2 = (int(proj_bezier_img[i + 1, 0]), int(proj_bezier_img[i + 1, 1]))
        cv2.line(centerline_draw_img_rgb, p1, p2, (0, 100, 255), 4)

    # Show
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    ax = axes.ravel()
    ax[0].remove()
    ax[1].remove()
    ax[2].remove()
    ax[3].remove()

    gs = GridSpec(2, 2, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cv2.cvtColor(centerline_draw_img_rgb, cv2.COLOR_BGR2RGB))
    ax0.set_title('Projected centerline')

    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    ax1.plot3D(curve_3D_gt[:, 0], curve_3D_gt[:, 1], curve_3D_gt[:, 2], color='#1f640a', linestyle='-', linewidth=2)
    ax1.plot3D(curve_3D_init[:, 0],
                curve_3D_init[:, 1],
                curve_3D_init[:, 2],
                color='#a64942',
                linestyle='--',
                linewidth=2)  ## green
    ax1.plot3D(curve_3D_opt[:, 0],
                curve_3D_opt[:, 1],
                curve_3D_opt[:, 2],
                color='#6F69AC',
                linestyle='-',
                linewidth=2)
    ax1.scatter(curve_3D_opt[-1, 0], curve_3D_opt[-1, 1], curve_3D_opt[-1, 2], marker='^', s=20,
                c=['#FFC069'])  ## yellow
    ax1.scatter(curve_3D_opt[0, 0], curve_3D_opt[0, 1], curve_3D_opt[0, 2], marker='o', s=20, c=['#FFC069'])

    ax1.quiver([0.0], [0.0], [0.0], [0.02], [0.0], [0.0], length=0.015, normalize=True, colors=['#911F27'])
    ax1.quiver([0.0], [0.0], [0.0], [0.00], [0.005], [0.0], length=0.003, normalize=True, colors=['#57CC99'])
    ax1.quiver([0.0], [0.0], [0.0], [0.00], [0.0], [0.04], length=0.04, normalize=True, colors=['#22577A'])

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.locator_params(nbins=4, axis='x')
    ax1.locator_params(nbins=4, axis='y')
    ax1.locator_params(nbins=4, axis='z')
    ax1.view_init(22, -26)
    ax1.set_title('gt/init/opt : green/red/blue')

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(saved_opt_history[1:, 0], color='#6F69AC', linestyle='-', linewidth=1)
    #ax2.text(-5, 60, 'Error = '+ str(error), fontsize = 15, position=(40,100))
    ax2.set_xlabel('Iterations and error=' + str(error))
    ax2.set_ylabel('Loss')

    # plt.tight_layout()
    plt.show()

def error_change(error):
    '''
    error ((a,b) tensor): a————record the error for every learning iteration
                          b————the error for the curve reconstructed
    '''
    plt.title("error results")
    for i in range(len(error)):
        x = range(i,len(error))
        y = error[i:][i]
        plt.plot(x,y)
    plt.show