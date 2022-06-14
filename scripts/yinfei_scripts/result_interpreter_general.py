import os
import numpy as np
import matplotlib.pyplot as plt

import path_settings
import experiment_setup
import identifier_conversions



n_mid_points = 1  ## This is assumed for now
n_iter = 20
n_data = 100
n_cases = 48  ## 16 x UN + 16 x IA0XX + 16 x IA1XX
loss_thresh_p3d = 5   ## Unit: mm
loss_thresh_p2d = 15  ## Unit: pixels

MASTER_params = np.zeros((n_iter + 2, 5, n_cases, n_data))
MASTER_p3d_poses = np.zeros((n_iter + 2, n_mid_points + 1, 3, n_cases, n_data))
MASTER_p2d_poses = np.zeros((n_iter + 2, n_mid_points + 1, 2, n_cases, n_data))

data_alias = 'D' + str(0).zfill(2)


for exp_name in experiment_setup.experiments:

    i = identifier_conversions.identifier_to_index[exp_name]

    method_dir = os.path.join(path_settings.results_dir, exp_name)

    for j in range(n_data):

        data_dir = os.path.join(method_dir, data_alias + '_' + str(j).zfill(4))
        params_report_path = os.path.join(data_dir, 'params.npy')
        p3d_report_path = os.path.join(data_dir, 'p3d_poses.npy')
        p2d_report_path = os.path.join(data_dir, 'p2d_poses.npy')

        params_report = np.load(params_report_path)
        p3d_report = np.load(p3d_report_path)
        p2d_report = np.load(p2d_report_path)

        MASTER_params[:, :, i, j] = params_report
        MASTER_p3d_poses[:, :, :, i, j] = p3d_report
        MASTER_p2d_poses[:, :, :, i, j] = p2d_report


MASTER_p3d_poses *= 1000  ## Convert meter to mm

print(MASTER_params.shape)
print(MASTER_p3d_poses.shape)
print(MASTER_p2d_poses.shape)


### Table 1 list 1: loss after last iteration, average by data
## Use target to subtract poses at each iteration
MASTER_p3d_pose_losses = MASTER_p3d_poses[-1, :, :, :, :] - MASTER_p3d_poses[0:-1, :, :, :, :]
MASTER_p2d_pose_losses = MASTER_p2d_poses[-1, :, :, :, :] - MASTER_p2d_poses[0:-1, :, :, :, :]

## Find the magnitude of loss
MASTER_p3d_pose_norm_losses = np.linalg.norm(MASTER_p3d_pose_losses, axis=2)
MASTER_p2d_pose_norm_losses = np.linalg.norm(MASTER_p2d_pose_losses, axis=2)

## Get the losses at tip (s=1)
MASTER_p3d_pose_norm_losses_tip = MASTER_p3d_pose_norm_losses[:, -1, :, :]
MASTER_p2d_pose_norm_losses_tip = MASTER_p2d_pose_norm_losses[:, -1, :, :]

## Average across data
MASTER_p3d_pose_norm_losses_tip_data_avg = np.mean(MASTER_p3d_pose_norm_losses_tip, axis=-1)
MASTER_p2d_pose_norm_losses_tip_data_avg = np.mean(MASTER_p2d_pose_norm_losses_tip, axis=-1)

## Get the losses after last iteration
MASTER_p3d_pose_norm_losses_tip_data_avg_final = MASTER_p3d_pose_norm_losses_tip_data_avg[-1, :]
MASTER_p2d_pose_norm_losses_tip_data_avg_final = MASTER_p2d_pose_norm_losses_tip_data_avg[-1, :]

table_1_average_loss_p3d = MASTER_p3d_pose_norm_losses_tip_data_avg_final
table_1_average_loss_p2d = MASTER_p2d_pose_norm_losses_tip_data_avg_final


### Table 1 list 2: number of iterations taken to get lower than some loss, average by data
table_1_iter_loss_thresh_p3d = np.ones(48) * -1
table_1_iter_loss_thresh_p2d = np.ones(48) * -1

print(MASTER_p3d_pose_norm_losses_tip.shape)


for i in range(n_cases):  ## loop through methods

    converge_iter_3d = []
    converge_iter_2d = []

    for j in range(n_data):  ## loop through data

        for k in range(n_iter):  ## loop through iterations
            if MASTER_p3d_pose_norm_losses_tip[k, i, j] < loss_thresh_p3d:
                converge_iter_3d.append(k)
                break

        for k in range(n_iter):  ## loop through iterations
            if MASTER_p2d_pose_norm_losses_tip[k, i, j] < loss_thresh_p2d:
                converge_iter_2d.append(k)
                break

    ## For each method, find the iteration to converge averaged on data
    if len(converge_iter_3d) > 0:
        table_1_iter_loss_thresh_p3d[i] = np.mean(np.array(converge_iter_3d))
    
    if len(converge_iter_2d) > 0:
        table_1_iter_loss_thresh_p2d[i] = np.mean(np.array(converge_iter_2d))
     

#for i in range(MASTER_p3d_pose_norm_losses_tip_data_avg.shape[1]):  ## loop through methods
#    for j in range(MASTER_p3d_pose_norm_losses_tip_data_avg.shape[0]):  ## loop through iterations
#        if MASTER_p3d_pose_norm_losses_tip_data_avg[j, i] < loss_thresh_p3d:
#            table_1_iter_loss_thresh_p3d[i] = j
#            break
#
#for i in range(MASTER_p2d_pose_norm_losses_tip_data_avg.shape[1]):
#    for j in range(MASTER_p2d_pose_norm_losses_tip_data_avg.shape[0]):
#        if MASTER_p2d_pose_norm_losses_tip_data_avg[j, i] < loss_thresh_p2d:
#            table_1_iter_loss_thresh_p2d[i] = j
#            break


### Table 1 list 3: percentage of data that each method achieves convergence (lower than some loss) by the end of its last iteration
MASTER_p3d_pose_norm_losses_tip_final = MASTER_p3d_pose_norm_losses_tip[-1, :, :]
MASTER_p2d_pose_norm_losses_tip_final = MASTER_p2d_pose_norm_losses_tip[-1, :, :]

table_1_convergence_percentage_p3d = np.zeros(48)
table_1_convergence_percentage_p2d = np.zeros(48)

for i in range(MASTER_p3d_pose_norm_losses_tip_final.shape[0]):
    for j in range(MASTER_p3d_pose_norm_losses_tip_final.shape[1]):
        if MASTER_p3d_pose_norm_losses_tip_final[i, j] < loss_thresh_p3d:
            table_1_convergence_percentage_p3d[i] += 1

for i in range(MASTER_p2d_pose_norm_losses_tip_final.shape[0]):
    for j in range(MASTER_p2d_pose_norm_losses_tip_final.shape[1]):
        if MASTER_p2d_pose_norm_losses_tip_final[i, j] < loss_thresh_p2d:
            table_1_convergence_percentage_p2d[i] += 1

table_1_convergence_percentage_p3d /= n_data
table_1_convergence_percentage_p2d /= n_data


### Table 1 combination
table_1 = np.zeros((48, 6))
table_1[:, 0] = table_1_average_loss_p2d
table_1[:, 1] = table_1_iter_loss_thresh_p2d
table_1[:, 2] = table_1_convergence_percentage_p2d * 100
table_1[:, 3] = table_1_average_loss_p3d
table_1[:, 4] = table_1_iter_loss_thresh_p3d
table_1[:, 5] = table_1_convergence_percentage_p3d * 100

np.savetxt(os.path.join(path_settings.results_dir, 'table_1.csv'), table_1, delimiter=',', fmt='%f')


### Table 1 2D
table_1_dof2 = np.zeros((12, 6))
table_1_dof2_identifiers = [
    'UN008', 'UN009', 'UN010', 'UN011', 'UN012', 'UN013', 'UN014', 'UN015',
    'IA008', 'IA009', 'IA010', 'IA011', 'IA012', 'IA013', 'IA014', 'IA015',
    'IA108', 'IA109', 'IA110', 'IA111', 'IA112', 'IA113', 'IA114', 'IA115']

for i, identifier in enumerate(table_1_dof2_identifiers):

    idx = identifier_conversions.identifier_to_index[identifier]

    if i % 2 == 0:
        table_1_dof2[i // 2, 0:3] = table_1[idx, 0:3]
    else:
        table_1_dof2[i // 2, 3:6] = table_1[idx, 0:3]

np.savetxt(os.path.join(path_settings.results_dir, 'table_1_2D.csv'), table_1_dof2, delimiter=',', fmt='%f')


### Table 1 3D
table_1_dof3 = np.zeros((12, 6))
table_1_dof3_identifiers = [
    'UN000', 'UN001', 'UN002', 'UN003', 'UN004', 'UN005', 'UN006', 'UN007',
    'IA000', 'IA001', 'IA002', 'IA003', 'IA004', 'IA005', 'IA006', 'IA007',
    'IA100', 'IA101', 'IA102', 'IA103', 'IA104', 'IA105', 'IA106', 'IA107']

for i, identifier in enumerate(table_1_dof3_identifiers):

    idx = identifier_conversions.identifier_to_index[identifier]

    if i % 2 == 0:
        table_1_dof3[i // 2, 0:3] = table_1[idx, 3:6]
    else:
        table_1_dof3[i // 2, 3:6] = table_1[idx, 3:6]

np.savetxt(os.path.join(path_settings.results_dir, 'table_1_3D.csv'), table_1_dof3, delimiter=',', fmt='%f')


### Figure set 1
plt.rcParams.update({'font.size': 12})
fontsize = 12
linewidth = 2

figure_set_1_p3d_data = np.zeros((n_iter + 1, n_cases))
figure_set_1_p2d_data = np.zeros((n_iter + 1, n_cases))

for i in range(n_cases):
    figure_set_1_p3d_data_temp = np.zeros((n_iter + 1, n_data))
    n_converge = 0

    for j in range(n_data):
        if MASTER_p3d_pose_norm_losses_tip[-1, i, j] < loss_thresh_p3d:
            figure_set_1_p3d_data_temp[:, n_converge] = MASTER_p3d_pose_norm_losses_tip[:, i, j]
            n_converge += 1

    figure_set_1_p3d_data[:, i] = np.mean(figure_set_1_p3d_data_temp[:, :n_converge], axis=1)


for i in range(n_cases):
    figure_set_1_p2d_data_temp = np.zeros((n_iter + 1, n_data))
    n_converge = 0
    
    for j in range(n_data):
        if MASTER_p2d_pose_norm_losses_tip[-1, i, j] < loss_thresh_p2d:
            figure_set_1_p2d_data_temp[:, n_converge] = MASTER_p2d_pose_norm_losses_tip[:, i, j]
            n_converge += 1

    figure_set_1_p2d_data[:, i] = np.mean(figure_set_1_p2d_data_temp[:, :n_converge], axis=1)



### 2D Tip Loss Evaluation
figure_set_1_L2_data = {
    '2D Tip Loss Evaluated on 2-DOF Cases Optimized for 2D Loss': [['UN008', 'UN010', 'IA008', 'IA010', 'IA108', 'IA110'], ['UN012', 'UN014', 'IA012', 'IA014', 'IA112', 'IA114']],
    '2D Tip Loss Evaluated on 2-DOF Cases Optimized for 3D Loss': [['UN000', 'UN002', 'IA000', 'IA002', 'IA100', 'IA102'], ['UN004', 'UN006', 'IA004', 'IA006', 'IA104', 'IA106']],
    '2D Tip Loss Evaluated on 3-DOF Cases Optimized for 2D Loss': [['UN009', 'UN011', 'IA009', 'IA011', 'IA109', 'IA111'], ['UN013', 'UN015', 'IA013', 'IA015', 'IA113', 'IA115']],
    '2D Tip Loss Evaluated on 3-DOF Cases Optimized for 3D Loss': [['UN001', 'UN003', 'IA001', 'IA003', 'IA101', 'IA103'], ['UN005', 'UN007', 'IA005', 'IA007', 'IA105', 'IA107']]
}

fig_set_1_L2, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, fig_data in enumerate(figure_set_1_L2_data):

    for identifier in figure_set_1_L2_data[fig_data][0]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p2d_data[:, idx], '--', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    for identifier in figure_set_1_L2_data[fig_data][1]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p2d_data[:, idx], '-', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    ax[i // 2, i % 2].set_yscale('log')
    ax[i // 2, i % 2].set_title(fig_data)
    ax[i // 2, i % 2].set_xlim([0, 20])
    ax[i // 2, i % 2].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax[i // 2, i % 2].set_xlabel('Iteration', fontsize=fontsize)
    ax[i // 2, i % 2].set_ylabel('Error (pixels)', fontsize=fontsize)
    ax[i // 2, i % 2].grid()

    if i % 2 != 0:
        ax[i // 2, i % 2].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

plt.tight_layout()
#plt.show()
#fig_set_1_L2.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L2.eps'))
#fig_set_1_L2.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L2.png'))



### 3D Tip Loss Evaluation
figure_set_1_L3_data = {
    '3D Tip Loss Evaluated on 2-DOF Cases Optimized for 2D Loss': [['UN008', 'UN010', 'IA008', 'IA010', 'IA108', 'IA110'], ['UN012', 'UN014', 'IA012', 'IA014', 'IA112', 'IA114']],
    '3D Tip Loss Evaluated on 2-DOF Cases Optimized for 3D Loss': [['UN000', 'UN002', 'IA000', 'IA002', 'IA100', 'IA102'], ['UN004', 'UN006', 'IA004', 'IA006', 'IA104', 'IA106']],
    '3D Tip Loss Evaluated on 3-DOF Cases Optimized for 2D Loss': [['UN009', 'UN011', 'IA009', 'IA011', 'IA109', 'IA111'], ['UN013', 'UN015', 'IA013', 'IA015', 'IA113', 'IA115']],
    '3D Tip Loss Evaluated on 3-DOF Cases Optimized for 3D Loss': [['UN001', 'UN003', 'IA001', 'IA003', 'IA101', 'IA103'], ['UN005', 'UN007', 'IA005', 'IA007', 'IA105', 'IA107']]
}

fig_set_1_L3, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, fig_data in enumerate(figure_set_1_L3_data):

    for identifier in figure_set_1_L3_data[fig_data][0]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p3d_data[:, idx], '--', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    for identifier in figure_set_1_L3_data[fig_data][1]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p3d_data[:, idx], '-', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    ax[i // 2, i % 2].set_yscale('log')
    ax[i // 2, i % 2].set_title(fig_data)
    ax[i // 2, i % 2].set_xlim([0, 20])
    ax[i // 2, i % 2].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax[i // 2, i % 2].set_xlabel('Iteration', fontsize=fontsize)
    ax[i // 2, i % 2].set_ylabel('Error (mm)', fontsize=fontsize)
    ax[i // 2, i % 2].grid()

    if i % 2 != 0:
        ax[i // 2, i % 2].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

plt.tight_layout()
#plt.show()
#fig_set_1_L3.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L3.eps'))
#fig_set_1_L3.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L3.png'))



### Tip Loss Evaluation (further split by same vs diff loss)
figure_set_1_L2_data_same_loss = {
    '2D Tip Loss of 2-DOF Cases Optimized for Tip Loss': [['UN008', 'IA008', 'IA108'], ['UN012', 'IA012', 'IA112']],
    '2D Tip Loss of 3-DOF Cases Optimized for Tip Loss': [['UN009', 'IA009', 'IA109'], ['UN013', 'IA013', 'IA113']],
    '2D Tip Loss of 2-DOF Cases Optimized for Shape Loss': [['UN010', 'IA010', 'IA110'], ['UN014', 'IA014', 'IA114']],
    '2D Tip Loss of 3-DOF Cases Optimized for Shape Loss': [['UN011', 'IA011', 'IA111'], ['UN015', 'IA015', 'IA115']]
}

fig_set_1_L2_same, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, fig_data in enumerate(figure_set_1_L2_data_same_loss):

    for identifier in figure_set_1_L2_data_same_loss[fig_data][0]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p2d_data[:, idx], '--', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    for identifier in figure_set_1_L2_data_same_loss[fig_data][1]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p2d_data[:, idx], '-', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    ax[i // 2, i % 2].set_yscale('log')
    #ax[i // 2, i % 2].set_title(fig_data)
    ax[i // 2, i % 2].set_xlim([0, 20])
    ax[i // 2, i % 2].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax[i // 2, i % 2].set_xlabel('Iteration', fontsize=fontsize)
    ax[i // 2, i % 2].set_ylabel('Error (pixels)', fontsize=fontsize)
    ax[i // 2, i % 2].grid()

    if i % 2 != 0:
        ax[i // 2, i % 2].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

plt.tight_layout()
#plt.show()
fig_set_1_L2_same.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L2_same.eps'))
fig_set_1_L2_same.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L2_same.png'))



figure_set_1_L3_data_same_loss = {
    '3D Tip Loss of 2-DOF Cases Optimized for Tip Loss': [['UN000', 'IA000', 'IA100'], ['UN004', 'IA004', 'IA104']],
    '3D Tip Loss of 3-DOF Cases Optimized for Tip Loss': [['UN001', 'IA001', 'IA101'], ['UN005', 'IA005', 'IA105']],
    '3D Tip Loss of 2-DOF Cases Optimized for Shape Loss': [['UN002', 'IA002', 'IA102'], ['UN006', 'IA006', 'IA106']],
    '3D Tip Loss of 3-DOF Cases Optimized for Shape Loss': [['UN003', 'IA003', 'IA103'], ['UN007', 'IA007', 'IA107']]
}

fig_set_1_L3_same, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, fig_data in enumerate(figure_set_1_L3_data_same_loss):

    for identifier in figure_set_1_L3_data_same_loss[fig_data][0]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p3d_data[:, idx], '--', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    for identifier in figure_set_1_L3_data_same_loss[fig_data][1]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p3d_data[:, idx], '-', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    ax[i // 2, i % 2].set_yscale('log')
    #ax[i // 2, i % 2].set_title(fig_data)
    ax[i // 2, i % 2].set_xlim([0, 20])
    ax[i // 2, i % 2].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax[i // 2, i % 2].set_xlabel('Iteration', fontsize=fontsize)
    ax[i // 2, i % 2].set_ylabel('Error (mm)', fontsize=fontsize)
    ax[i // 2, i % 2].grid()

    if i % 2 != 0:
        ax[i // 2, i % 2].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

plt.tight_layout()
#plt.show()
fig_set_1_L3_same.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L3_same.eps'))
fig_set_1_L3_same.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L3_same.png'))



figure_set_1_L2_data_diff_loss = {
    '2D Tip Loss Evaluated on 2-DOF Cases Optimized for 3D Tip Loss': [['UN000', 'IA000', 'IA100'], ['UN004', 'IA004', 'IA104']],
    '2D Tip Loss Evaluated on 3-DOF Cases Optimized for 3D Tip Loss': [['UN001', 'IA001', 'IA101'], ['UN005', 'IA005', 'IA105']],
    '2D Tip Loss Evaluated on 2-DOF Cases Optimized for 3D Shape Loss': [['UN002', 'IA002', 'IA102'], ['UN006', 'IA006', 'IA106']],
    '2D Tip Loss Evaluated on 3-DOF Cases Optimized for 3D Shape Loss': [['UN003', 'IA003', 'IA103'], ['UN007', 'IA007', 'IA107']]
}

fig_set_1_L2_diff, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, fig_data in enumerate(figure_set_1_L2_data_diff_loss):

    for identifier in figure_set_1_L2_data_diff_loss[fig_data][0]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p2d_data[:, idx], '--', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    for identifier in figure_set_1_L2_data_diff_loss[fig_data][1]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p2d_data[:, idx], '-', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    ax[i // 2, i % 2].set_yscale('log')
    ax[i // 2, i % 2].set_title(fig_data)
    ax[i // 2, i % 2].set_xlim([0, 20])
    ax[i // 2, i % 2].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax[i // 2, i % 2].set_xlabel('Iteration', fontsize=fontsize)
    ax[i // 2, i % 2].set_ylabel('Error (pixels)', fontsize=fontsize)
    ax[i // 2, i % 2].grid()

    if i % 2 != 0:
        ax[i // 2, i % 2].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

plt.tight_layout()
#plt.show()
#fig_set_1_L2_diff.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L2_diff.eps'))
#fig_set_1_L2_diff.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L2_diff.png'))



figure_set_1_L3_data_diff_loss = {
    '3D Tip Loss Evaluated on 2-DOF Cases Optimized for 2D Tip Loss': [['UN008', 'IA008', 'IA108'], ['UN012', 'IA012', 'IA112']],
    '3D Tip Loss Evaluated on 3-DOF Cases Optimized for 2D Tip Loss': [['UN009', 'IA009', 'IA109'], ['UN013', 'IA013', 'IA113']],
    '3D Tip Loss Evaluated on 2-DOF Cases Optimized for 2D Shape Loss': [['UN010', 'IA010', 'IA110'], ['UN014', 'IA014', 'IA114']],
    '3D Tip Loss Evaluated on 3-DOF Cases Optimized for 2D Shape Loss': [['UN011', 'IA011', 'IA111'], ['UN015', 'IA015', 'IA115']]
}

fig_set_1_L3_diff, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, fig_data in enumerate(figure_set_1_L3_data_diff_loss):

    for identifier in figure_set_1_L3_data_diff_loss[fig_data][0]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p3d_data[:, idx], '--', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    for identifier in figure_set_1_L3_data_diff_loss[fig_data][1]:
        idx = identifier_conversions.identifier_to_index[identifier]
        ax[i // 2, i % 2].plot(figure_set_1_p3d_data[:, idx], '-', linewidth=linewidth, label=identifier_conversions.identifier_to_description_no_dof[identifier])

    ax[i // 2, i % 2].set_yscale('log')
    ax[i // 2, i % 2].set_title(fig_data)
    ax[i // 2, i % 2].set_xlim([0, 20])
    ax[i // 2, i % 2].set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax[i // 2, i % 2].set_xlabel('Iteration', fontsize=fontsize)
    ax[i // 2, i % 2].set_ylabel('Error (mm)', fontsize=fontsize)
    ax[i // 2, i % 2].grid()

    if i % 2 != 0:
        ax[i // 2, i % 2].legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

plt.tight_layout()
#plt.show()
#fig_set_1_L3_diff.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L3_diff.eps'))
#fig_set_1_L3_diff.savefig(os.path.join(path_settings.results_dir, 'figure_set_1_L3_diff.png'))
