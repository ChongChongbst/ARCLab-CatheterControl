import os
import subprocess
import numpy as np
import path_settings



class BezierSet:

    def __init__(self, n):
        """
        Args:
            n (int): number of Bezier curves to render
        """
        self.specs = np.zeros((n, 4, 3))
        self.count = 0


    def enter_spec(self, p_start, p_end, c1, c2):
        """
        Fill the specs 3D array, which represents all the Bezier curves to render

        Args:
            p_start ((3,) numpy array): start point of Bezier curve
            p_end ((3,) numpy array): end point of Bezier curve
            c1 ((3,) numpy array): control point 1 of Bezier curve
            c2 ((3,) numpy array): control point 2 of Bezier curve
        
        Note:
            c = (p_mid - (p_start / 4) - (p_end / 4)) * 2
            c1 = 4 / 3 * p_mid - 1 / 3 * p_end
            c2 = 4 / 3 * p_mid - 1 / 3 * p_start
            where p_mid is on the curve; c, c1, and c2 are not on the curve
        """
        self.specs[self.count, :] = np.array([p_start, p_end, c1, c2])
        self.count += 1


    def print_specs(self):
        """
        Print the specs 3D array
        """
        print('Bezier Specs (shape = ' + str(self.specs.shape) + '): ')
        print(self.specs)
        print('')


    def write_specs(self, specs_path):
        """
        Write the specs 3D array to a specified path

        Args:
            specs_path (path string to npy file): specified path to write the specs 3D array  
        """
        self.specs_path = specs_path
        np.save(specs_path, self.specs)


    def render(self, img_save_path, target_specs_path=None, viewpoint_mode=1, transparent_mode=0):
        """
        Call Blender to render the Bezier curves

        Args:
            img_save_path (path string to png file): path to save the rendered image
            target_specs_path (path stirng to npy file): path to an existing target specs file
            viewpoint_mode (1 or 2): camera view of rendered image, 1 for endoscopic view, 2 for side view
            transparent_mode (0 or 1): whether to make the background transparent for the rendered image, 0 for not transparent, 1 for transparent
        """
        os.chdir(path_settings.blender_dir)

        if target_specs_path:
            subprocess.run([
                './blender',
                '-b',
                '-P', path_settings.bezier_render_script,
                '--',
                '--specs_path', self.specs_path,
                '--save_path', img_save_path,
                '--viewpoint_mode', str(viewpoint_mode),
                '--target_specs_path', target_specs_path,
                '--transparent_mode', str(transparent_mode)])

        else:
            subprocess.run([
                './blender',
                '-b',
                '-P', path_settings.bezier_render_script,
                '--',
                '--specs_path', self.specs_path,
                '--save_path', img_save_path,
                '--viewpoint_mode', str(viewpoint_mode),
                '--target_specs_path', '',
                '--transparent_mode', str(transparent_mode)])
