### Usage


##### Files 
- reconst_sim_opt2pts.py : optimization for [PC, P1], by assuming [P0] is a constant point
- reconst_sim_opt3pts.py : optimization for [P0, PC, P1] all 3 bezier points
- it is better to use reconst_sim_opt2pts.py, which normally converged to ground truth
- reconst_sim_opt3pts.py may not converged to ground truth, due to loosing of depth information

##### Parameters
| para              | Description |
| -----------       | ----------- |
| img_path          | image path       |
| curve_length_gt   | ground truth bezier curve length from P0 to P1        |
| para_gt           | ground truth bezier points : [P0, PC, P1]        |
| para_init         | initialized bezier points : [P0, PC, P1]        |
| loss_weight       | weights for three different loss : centerline/tip/curvelength |
| total_itr         | total itr for gradients optimization |


##### Output
| para                           | Description |
| -----------                    | ----------- |
| class.para                     | optimized bezier points : [P0, PC, P1]   |
| class.saved_opt_history        | each row : [loss of iter, P0, PC, P1]    |

