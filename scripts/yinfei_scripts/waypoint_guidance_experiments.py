import os
import numpy as np

import path_settings
import contour_tracer
from experiment_setup import experiments
from simulation_experiment import SimulationExperiment



### Universal parameters
p_0 = np.array([2e-2, 2e-3, 0])
r = 0.01
n_iter = 10
noise_percentage = 0.25
ux_init = 0.00001
uy_init = 0.00001
l_init = 0.2

identifiers_of_interest = ['IA009']
image_names = ['tumor4595_mask']

#identifiers_of_interest = ['UN008', 'UN009', 'IA008', 'IA009', 'IA108', 'IA109', 'UN012', 'UN013', 'IA012', 'IA013', 'IA112', 'IA113']
#image_names = ['circle', 'rectangle', 'heart', 'tumor4595_mask']


for image_name in image_names:

    image_path = os.path.join(path_settings.contour_images_dir, image_name + '.png')

    ## Get waypoints from image contour
    ct = contour_tracer.ContourTracer(image_path)
    waypoints_2d = ct.trace_contour()
    print('Number of waypoints: ', waypoints_2d.shape[0])

    ## Draw contour and resized images
    contour_image_path = image_path[:-4] + '_contour.png'
    resized_image_path = image_path[:-4] + '_resized.png'
    ct.draw_contour(contour_image_path)
    ct.draw_resized_image(resized_image_path)

    ## Sample waypoints
    selected_indices = np.arange(0, waypoints_2d.shape[0], 50)
    waypoints_2d_selected = waypoints_2d[selected_indices, :]

    n_data = waypoints_2d_selected.shape[0]
    print('Number of sampled waypoints: ', n_data)


    for identifier in identifiers_of_interest:

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Running waypoint guidance experiment ', identifier)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        exp = experiments[identifier]

        dof = exp['dof']
        loss_2d = exp['loss_2d']
        tip_loss = exp['tip_loss']
        use_reconstruction = exp['use_reconstruction']
        interspace = exp['interspace']
        #viewpoint_mode = exp['viewpoint_mode']
        viewpoint_mode = 3
        damping_weights = exp['damping_weights']
        n_mid_points = exp['n_mid_points']

        if use_reconstruction:
            render_mode = 2
        else:
            render_mode = 1

        method_dir = os.path.join(path_settings.results_dir, identifier)

        if not os.path.isdir(method_dir):
            os.mkdir(method_dir)
        
        data_dir_outer = os.path.join(method_dir, image_name)

        if not os.path.isdir(data_dir_outer):
            os.mkdir(data_dir_outer)

        ux_old = None
        uy_old = None
        l_old = None

        for i in range(n_data):

            data_dir = os.path.join(data_dir_outer, str(i).zfill(4))
            images_save_dir = os.path.join(data_dir, 'images')
            cc_specs_save_dir = os.path.join(data_dir, 'cc_specs')
            params_report_path = os.path.join(data_dir, 'params.npy')
            p3d_report_path = os.path.join(data_dir, 'p3d_poses.npy')
            p2d_report_path = os.path.join(data_dir, 'p2d_poses.npy')

            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
                os.mkdir(images_save_dir)
                os.mkdir(cc_specs_save_dir)

            if i == 0:
                ux = ux_init
                uy = uy_init
                l = l_init
            else:
                ux = ux_old
                uy = uy_old

                if dof == 3:
                    l = l_old
                else:
                    l = l_init

            x_target = waypoints_2d_selected[i, 0]
            y_target = waypoints_2d_selected[i, 1]

            sim_exp = SimulationExperiment(dof, loss_2d, tip_loss, use_reconstruction, interspace, viewpoint_mode, damping_weights, noise_percentage, n_iter, render_mode)
            sim_exp.set_paths(images_save_dir, cc_specs_save_dir, params_report_path, p3d_report_path, p2d_report_path)
            sim_exp.set_general_parameters(p_0, r, n_mid_points, l)
            sim_exp.set_2d_pos_parameters(ux, uy, x_target, y_target, l)
            
            params = sim_exp.execute()

            ux_old = params[-2, 0]
            uy_old = params[-2, 1]
            l_old = params[-2, 2]
