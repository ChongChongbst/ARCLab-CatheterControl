import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import path_settings
import identifier_conversions



n_iter = 10
n_trials = 10

identifiers_of_interest = [
    'UN008', 'IA008', 'IA108', 
    'UN009', 'IA009', 'IA109',
    'UN012', 'IA012', 'IA112',
    'UN013', 'IA013', 'IA113']


## Generate cast-net data
data_alias = 'CN0'
spacing = 20
x_targets = np.arange(5, 640 - 5, spacing)
y_targets = np.arange(5, 480 - 5, spacing)

### Figure set 4
plt.rcParams.update({'font.size': 12})

fig_set_4, ax = plt.subplots(4, 3, figsize = (20, 20))

for k, identifier in enumerate(identifiers_of_interest):

    method_dir = os.path.join(path_settings.results_dir, identifier)    
    data_dir_outer = os.path.join(method_dir, data_alias)

    castnet_result_p2d = np.zeros((len(y_targets), len(x_targets)))

    for i, x_target in enumerate(x_targets):
        for j, y_target in enumerate(y_targets):

            p2d_diffs = []

            for l in range(n_trials):

                data_dir = os.path.join(data_dir_outer, str(x_target).zfill(4) + '_' + str(y_target).zfill(4) + '_' + str(l).zfill(2))
                ## Record p2d final diff
                p2d_report_path = os.path.join(data_dir, 'p2d_poses.npy')
                p2d_report = np.load(p2d_report_path)
                p2d_target = p2d_report[-1, -1, :]
                p2d_result = p2d_report[n_iter, -1, :]
                p2d_diff = np.linalg.norm(p2d_target - p2d_result, 2)

                p2d_diffs.append(p2d_diff)

            castnet_result_p2d[j, i] = np.mean(np.array(p2d_diffs))

    castnet_result_p2d[castnet_result_p2d > 1e3] = 1e3
    #castnet_result_p2d[castnet_result_p2d < 1e-3] = 1e-3
    castnet_result_p2d_interpolated = cv2.resize(castnet_result_p2d, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)

    img = ax[k // 3, k % 3].imshow(castnet_result_p2d_interpolated, origin='upper')
    #img = ax[k // 3, k % 3].imshow(castnet_result_p2d_interpolated, origin='upper', norm=matplotlib.colors.LogNorm(vmin=1e-10, vmax=1e10))
    ax[k // 3, k % 3].set_title(identifier_conversions.identifier_to_description[identifier])
    plt.clim(0, 100)
    cbar = plt.colorbar(img, ax=ax[k // 3, k % 3])
    cbar.ax.set_ylabel('Error (pixels)')

plt.tight_layout()
#plt.show()
fig_set_4.savefig(os.path.join(path_settings.results_dir, 'figure_set_4.eps'))
fig_set_4.savefig(os.path.join(path_settings.results_dir, 'figure_set_4.png'))
