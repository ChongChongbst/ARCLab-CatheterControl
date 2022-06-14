import os
import numpy as np

import path_settings
from experiment_setup import experiments
from real_robot_experiment import RealRobotExperiment



### Universal parameters
p_0 = np.array([2e-2, 2e-3, 0])
r = 0.01
n_iter = 10
n_trials = 10
noise_percentage = 0.25
ux_init = 0.00001
uy_init = 0.00001
l_init = 0.2

# identifiers_of_interest = ['UN012', 'UN013', 'IA012', 'IA013', 'IA112', 'IA113']

identifier = 'UN012'
data_alias = 'RR01'

## set_targets
x_target = 50
y_target = 100


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Running real experiment experiment for ', identifier)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

exp = experiments[identifier]
dof = exp['dof']
loss_2d = exp['loss_2d']
tip_loss = exp['tip_loss']
use_reconstruction = exp['use_reconstruction']

if not use_reconstruction:
    print('[ERROR] Real robot experiments requires reconstruction')
    exit()

interspace = exp['interspace']
viewpoint_mode = exp['viewpoint_mode']
damping_weights = exp['damping_weights']
n_mid_points = exp['n_mid_points']

render_mode = 2

method_dir = os.path.join(path_settings.results_dir, identifier)
if not os.path.isdir(method_dir):
    os.mkdir(method_dir)

data_dir_outer = os.path.join(method_dir, data_alias)
if not os.path.isdir(data_dir_outer):
    os.mkdir(data_dir_outer)

data_dir = os.path.join(data_dir_outer, str(x_target).zfill(4) + '_' + str(y_target).zfill(4))
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

sim_exp = RealRobotExperiment(dof, loss_2d, tip_loss, interspace, viewpoint_mode, damping_weights, noise_percentage, n_iter, render_mode)
sim_exp.set_paths(images_save_dir, cc_specs_save_dir, params_report_path, p3d_report_path, p2d_report_path)
sim_exp.set_general_parameters(p_0, r, n_mid_points, l)
sim_exp.set_2d_pos_parameters(ux, uy, x_target, y_target, l)
sim_exp.execute()

