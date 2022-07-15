import os
import numpy as np

import camera_settings
import path_settings
from simulation_experiment import SimulationExperiment
from data_generation import DataGeneration
from experiment_setup import experiments



### Universal parameters
p_0 = np.array([2e-2, 2e-3, 0])
r = 0.01
n_iter = 20
n_data = 100
noise_percentage = 0.25
ux_init = 0.00001
uy_init = 0.00001
l_init = 0.2


### Target parameter data generation
data_alias = 'D' + str(0).zfill(2)
data_save_path = os.path.join(path_settings.target_parameters_dir, data_alias + '.npy')
s_list = [0.5, 1]

data_gen = DataGeneration(n_data, p_0, r, l_init, s_list, data_save_path)
data_gen.set_target_ranges(-0.005, 0.005, -0.005, 0.005, 0.1, 0.5)
data_gen.set_camera_params(camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y, camera_settings.image_size_x, camera_settings.image_size_y, camera_settings.extrinsics)
data_gen.generate_data()
target_parameters = np.load(data_save_path)


### Test setup for all methods using general dataset

error_reports = []

for exp_name in experiments:

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Running experiment ', exp_name)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    exp = experiments[exp_name]

    dof = exp['dof']
    loss_2d = exp['loss_2d']
    tip_loss = exp['tip_loss']
    use_reconstruction = exp['use_reconstruction']
    interspace = exp['interspace']
    viewpoint_mode = exp['viewpoint_mode']
    damping_weights = exp['damping_weights']
    n_mid_points = exp['n_mid_points']
    
    if use_reconstruction:
        render_mode = 2
    else:
        render_mode = 1

    method_dir = os.path.join(path_settings.results_dir, exp_name)

    if not os.path.isdir(method_dir):
        os.mkdir(method_dir)

    for i in range(n_data):
        print('Executing experiment ' + exp_name + ' for data ' + str(i) + ': ', target_parameters[i, :])

        try:

            data_dir = os.path.join(method_dir, data_alias + '_' + str(i).zfill(4))

            images_save_dir = os.path.join(data_dir, 'images')
            cc_specs_save_dir = os.path.join(data_dir, 'cc_specs')
            params_report_path = os.path.join(data_dir, 'params.npy')
            p3d_report_path = os.path.join(data_dir, 'p3d_poses.npy')
            p2d_report_path = os.path.join(data_dir, 'p2d_poses.npy')

            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
                os.mkdir(images_save_dir)
                os.mkdir(cc_specs_save_dir)

            ux = ux_init
            uy = uy_init
            l = l_init

            ux_target = target_parameters[i, 0]
            uy_target = target_parameters[i, 1]
            l_target = target_parameters[i, 2]

            sim_exp = SimulationExperiment(dof, loss_2d, tip_loss, use_reconstruction, interspace, viewpoint_mode, damping_weights, noise_percentage, n_iter, render_mode)
            sim_exp.set_paths(images_save_dir, cc_specs_save_dir, params_report_path, p3d_report_path, p2d_report_path)
            sim_exp.set_general_parameters(p_0, r, n_miad_points, l)

            if dof == 2:
                sim_exp.set_2dof_parameters(ux, uy, ux_target, uy_target)
            elif dof == 3:
                sim_exp.set_3dof_parameters(ux, uy, ux_target, uy_target, l_target)
            else:
                print('[ERROR] DOF not defined')

            sim_exp.execute()
        
        except:
            error_reports.append('Experiment ' + exp_name + ' failed for data ' + str(i))
        
print(error_reports)
