import os
import numpy as np
import cv2

import Chong_scripts.find_curvature as tc
import camera_settings
from cc_catheter import CCCatheter
 
### Universal parameters
p_0 = np.array([2e-2, 2e-3, 0])
r = 0.01
n_iter = 10
n_trials = 10
noise_percentage = 0.25
ux_init = 0.00001
uy_init = 0.00001
l_init = 0.2

catheter = CCCatheter(p_0, l, r, loss_2d, tip_loss, n_mid_points, n_iter, verbose=0)