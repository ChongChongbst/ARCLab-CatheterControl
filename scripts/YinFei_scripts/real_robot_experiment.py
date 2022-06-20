import os
import torch
import sys
import path_settings
sys.path.insert(0, path_settings.scripts_dir) 

import camera_settings
from cc_catheter import CCCatheter
from reconstruction_scripts.reconst_sim_opt2pts import reconstructCurve



class RealRobotExperiment:

    def __init__(self, dof, loss_2d, tip_loss, interspace, viewpoint_mode, damping_weights, noise_percentage, n_iter, render_mode):
        """
        Args:
            dof (1, 2, or 3): DoF of control (1 DoF is not fully implemented currently)
            loss_2d (bool): whether to use 2D loss
            tip_loss (bool): whether to use tip loss
            interspace (0, 1, or 2): interspace of control, 0 for unispace, 1 for Bezier interspace
                with (theta, phi) parameterization, 2 for Bezier interspace with (ux, uy) parameterization
            viewpoint_mode (1 or 2): camera view of rendered image, 1 for endoscopic view, 2 for side view
            damping_weights (list of 3 floats): n-th term corresponds to the damping weight of the n-th DoF control feedback
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
            n_iter (int): number of total iteration of optimization
            render_mode (0, 1, or 2): 0 for rendering no image, 1 for only rendering the image after
                the last iteration, 2 for rendering all image 
        """
        self.dof = dof
        self.loss_2d = loss_2d
        self.tip_loss = tip_loss
        self.interspace = interspace
        self.viewpoint_mode = viewpoint_mode
        self.damping_weights = damping_weights
        self.noise_percentage = noise_percentage
        self.n_iter = n_iter
        self.render_mode = render_mode

        self.use_2d_pos_target = False


    def set_paths(self, images_save_dir, cc_specs_save_dir, params_report_path, p3d_report_path, p2d_report_path):
        """
        Args:
            images_save_dir (path string to directory): directory to save rendered images
            cc_specs_save_dir (path string to directory): directory to save constant curvature specs
            params_report_path (path string to npy file): self.params is a (n_iter + 2, 5) numpy array
                the first row records the initial parameters;
                the last row records the target parameters;
                and the intermediate rows record the parameters throughout the iterations.
                The 5 columns records the ux, uy, l, theta, phi parameters.
                If some parameters are not applicable for current method, they are left as 0
            p3d_report_path (path string to npy file)
            p2d_report_path (path string to npy file)
        """
        self.images_save_dir = images_save_dir
        self.cc_specs_save_dir = cc_specs_save_dir
        self.params_report_path = params_report_path
        self.p3d_report_path = p3d_report_path        
        self.p2d_report_path = p2d_report_path
        
        
    def set_general_parameters(self, p_0, r, n_mid_points, l):
        """
        Args:
            p_0 ((3,) numpy array): start point of catheter
            r (float): cross section radius of catheter
            n_mid_points (int): number of middle control points
            l (float): length of catheter
        """
        self.p_0 = p_0
        self.r = r
        self.l = l
        self.n_mid_points = n_mid_points


    def set_1dof_parameters(self, u, phi, u_target):
        """
        Set parameters for 1DoF (not fully implemented currently)

        Args:
            u (float): tendon length (responsible for catheter bending)
            phi (radians as float): phi parameter
            u_target (float): target tendon length (responsible for catheter bending)
        """
        self.u = u
        self.phi = phi
        self.u_target = u_target


    def set_2dof_parameters(self, ux, uy, ux_target, uy_target):
        """
        Set parameters for 2DoF

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            ux_target (int): target of 1st pair of tendon length
            uy_target (int): target of 2nd pair of tendon length
        """
        self.ux = ux
        self.uy = uy
        self.ux_target = ux_target
        self.uy_target = uy_target


    def set_3dof_parameters(self, ux, uy, ux_target, uy_target, l_target):
        """
        Set parameters for 3DoF

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            ux_target (int): target of 1st pair of tendon length
            uy_target (int): target of 2nd pair of tendon length
            l_target (float): target length of bending portion of the catheter (responsible for insertion)
        """
        self.ux = ux
        self.uy = uy
        self.ux_target = ux_target
        self.uy_target = uy_target
        self.l_target = l_target

    
    def set_2d_pos_parameters(self, ux, uy, x_target, y_target, l=0):
        """
        Set parameters for 2D loss

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            x_target (int): horizontal target pixel location of end effector
            y_target (int): vertical target pixel location of end effector
            l (float): length of bending portion of the catheter (responsible for insertion)
        """
        if not (self.loss_2d and self.tip_loss):
            print('[ERROR] Setting 2D position target is not compatible with non 2D tip loss')
            exit()

        self.ux = ux
        self.uy = uy
        self.x_target = x_target
        self.y_target = y_target

        if self.dof == 3:
            self.l = l

        self.use_2d_pos_target = True


    def execute(self):
        """
        Run main pipeline of inverse Jacobian control

        Returns:
            params ((n_iter + 2, 5) numpy array): 
                the first row records the initial parameters;
                the last row records the target parameters;
                and the intermediate rows record the parameters throughout the iterations.
                The 5 columns records the ux, uy, l, theta, phi parameters.
                If some parameters are not applicable for current method, they are left as 0
        """
        catheter = CCCatheter(self.p_0, self.l, self.r, self.loss_2d, self.tip_loss, self.n_mid_points, self.n_iter, verbose=0)
        catheter.set_weight_matrix(self.damping_weights[0], self.damping_weights[1], self.damping_weights[2])

        if self.use_2d_pos_target:
            assert self.ux
            assert self.uy
            assert self.x_target
            assert self.y_target

            if self.dof == 2:
                catheter.set_2dof_params(self.ux, self.uy)
            elif self.dof == 3:
                catheter.set_3dof_params(self.ux, self.uy, self.l)
            else:
                print('[ERROR] DOF invalid')

            catheter.set_2d_targets([0, self.x_target], [0, self.y_target])

        else:
            if self.dof == 1:
                assert self.u
                assert self.phi
                assert self.u_target

                catheter.set_1dof_params(self.phi, self.u)
                catheter.set_1dof_targets(self.u_target)

            elif self.dof == 2:
                assert self.ux
                assert self.uy
                assert self.ux_target
                assert self.uy_target

                catheter.set_2dof_params(self.ux, self.uy)
                catheter.set_2dof_targets(self.ux_target, self.uy_target)

            elif self.dof == 3:
                assert self.ux
                assert self.uy
                assert self.ux_target
                assert self.uy_target
                assert self.l_target

                catheter.set_3dof_params(self.ux, self.uy, self.l)
                catheter.set_3dof_targets(self.ux_target, self.uy_target, self.l_target)

            else:
                print('[ERROR] DOF invalid')

        catheter.set_camera_params(camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y, camera_settings.image_size_x, camera_settings.image_size_y, camera_settings.extrinsics)
        catheter.calculate_cc_points(init=True)
        catheter.convert_cc_points_to_2d(init=True)
        catheter.calculate_beziers_control_points()

        if not self.use_2d_pos_target:
            catheter.calculate_cc_points(target=True)
            catheter.convert_cc_points_to_2d(target=True)
        
        cc_specs_path = os.path.join(self.cc_specs_save_dir , '000.npy')
        image_save_path = os.path.join(self.images_save_dir, '000.png')

        bezier_specs_old = catheter.calculate_bezier_specs()
        target_specs_path = None

        if self.render_mode == 2: 
            catheter.render_beziers(cc_specs_path, image_save_path, target_specs_path, self.viewpoint_mode, transparent_mode=0)

        for i in range(self.n_iter):
            print('------------------------- Start of Iteration ' + str(i) + ' -------------------------')

            if self.dof == 1:
                catheter.update_1dof_params(i, self.noise_percentage)

            elif self.dof == 2:
                if self.interspace == 0:
                    catheter.update_2dof_params(i, self.noise_percentage)
                elif self.interspace == 1:
                    catheter.update_2dof_params_bezier_interspace_ux_uy(i, self.noise_percentage)
                elif self.interspace == 2:
                    catheter.update_2dof_params_bezier_interspace_theta_phi(i, self.noise_percentage)

            else:
                if self.interspace == 0:
                    catheter.update_3dof_params(i, self.noise_percentage)
                elif self.interspace == 1:
                    catheter.update_3dof_params_bezier_interspace_ux_uy(i, self.noise_percentage)
                elif self.interspace == 2:
                    catheter.update_3dof_params_bezier_interspace_theta_phi(i, self.noise_percentage)

            ## Interact with the real catheter here
            updated_params = catheter.get_params()
            print('Updated params for iteration ', str(i), ': ', updated_params[i + 1, :])
            input("Press Enter to continue...")


            catheter.calculate_cc_points(i)
            catheter.convert_cc_points_to_2d(i)
            catheter.calculate_beziers_control_points()                

            cc_specs_path = os.path.join(self.cc_specs_save_dir, str(i + 1).zfill(3) + '.npy')
            rendered_image_save_path = os.path.join(self.images_save_dir, str(i + 1).zfill(3) + '.png')

            ## FIXME Change this !!!
            captured_image_read_path = rendered_image_save_path

            if self.render_mode > 0:
                if i == (self.n_iter - 1):
                    catheter.render_beziers(cc_specs_path, rendered_image_save_path, target_specs_path, self.viewpoint_mode, transparent_mode=1)
                elif self.render_mode == 2:
                    catheter.render_beziers(cc_specs_path, rendered_image_save_path, target_specs_path, self.viewpoint_mode, transparent_mode=0)

            ### Fei's reconstruction

            ## Get Bezier specs of current curve
            bezier_specs = catheter.calculate_bezier_specs()
            bezier_specs_torch = torch.tensor(bezier_specs.flatten(), dtype=torch.float)
            bezier_specs_init_torch = torch.tensor(bezier_specs_old.flatten(), dtype=torch.float, requires_grad=True)

            loss_weight = torch.tensor([1.0, 1.0, 1.0])
            p_0 = torch.tensor(catheter.p_0)

            ## Detect actual bezier
            bezier_reconstruction = reconstructCurve(captured_image_read_path, catheter.l, p_0, bezier_specs_torch, bezier_specs_init_torch, loss_weight, total_itr=50)
            bezier_reconstruction.getOptimize(None, p_0)
            #bezier_reconstruction.plotProjCenterline()

            ## Convert actual bezier to cc
            optimized_bezier_specs = bezier_reconstruction.para.detach().numpy().reshape((2, 3))
            catheter.convert_bezier_to_cc(optimized_bezier_specs)
            bezier_specs_old = bezier_specs
            catheter.convert_cc_points_to_2d(i)

            print('-------------------------- End of Iteration ' + str(i) + ' --------------------------')

        catheter.write_reports(self.params_report_path, self.p3d_report_path, self.p2d_report_path)

        return catheter.get_params()
