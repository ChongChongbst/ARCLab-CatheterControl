import os
import bpy
import argparse
import numpy as np

from math import *
from mathutils import Euler, Matrix, Quaternion, Vector
from bpy import context, data, ops
 


parser = argparse.ArgumentParser()

# get all script args
_, all_arguments = parser.parse_known_args()
double_dash_index = all_arguments.index('--')
script_args = all_arguments[double_dash_index + 1: ]
parser.add_argument('specs_path', help='Path to an npy file. Specs of the Bezier curves to be rendered.')
parser.add_argument('save_path', help='Path to save the rendered images. Assume PNG.')
parser.add_argument('viewpoint_mode', help='Setting of viewpoint option. 1 for catheter POV and 2 for side view.', default=1)
parser.add_argument('target_specs_path', help='Path to an npy file. Specs of the target points to be rendered.', default=None)
parser.add_argument('transparent_mode', help='Whether or not to make rendering background transparent', default=0)
args, _ = parser.parse_known_args(script_args)


bezier_specs = np.load(args.specs_path)
n_beziers = bezier_specs.shape[0]


ops.object.select_all(action='SELECT')
ops.object.delete(use_global=False)


# create Material
mat_blue = data.materials.new(name="Material_blue")
mat_blue.diffuse_color = (0.0553055, 0.0761522, 0.263268, 1)

mat_green = data.materials.new(name="Material_green")
mat_green.diffuse_color = (0.036845, 0.263268, 0.120192, 1)

mat_red = data.materials.new(name="Material_red")
mat_red.diffuse_color = (1, 0, 0, 1)


for i in range(n_beziers):

    # Create curve and cache reference.
    ops.curve.primitive_bezier_curve_add(enter_editmode=False, align='WORLD', location=(0, 0, 0))

    curve = context.active_object
    curve.name = 'catheter_curve_' + str(i)
    curve.data.resolution_u = 50

    bez_points = curve.data.splines[0].bezier_points

    p_start = bezier_specs[i, 0, :]
    p_end = bezier_specs[i, 1, :]
    c1 = bezier_specs[i, 2, :]
    c2 = bezier_specs[i, 3, :]

    # Left point.
    bez_points[0].co = p_start
    bez_points[0].handle_left_type = 'FREE'
    bez_points[0].handle_right_type = 'FREE'
    bez_points[0].handle_left = Vector((1.0, 0.0, 0.0))  # not needed
    bez_points[0].handle_right = c1

    # Top-middle point.
    bez_points[1].co = p_end
    bez_points[1].handle_left_type = 'FREE'
    bez_points[1].handle_right_type = 'FREE'
    bez_points[1].handle_left = c2
    bez_points[1].handle_right = Vector((1.0, 0.0, 0.0))  # not needed

    curve.data.bevel_depth = 0.0015
    curve.data.bevel_resolution = 10
    curve.data.materials.append(mat_blue)


## Target Visualization
if len(args.target_specs_path):
    target_specs = np.load(args.target_specs_path)
    n_targets = target_specs.shape[0]

    for i in range(n_targets):
        ops.mesh.primitive_uv_sphere_add(radius=0.002, location=target_specs[i, :])   
        ops.object.shade_smooth()

        if i == 0:
            data.objects['Sphere'].data.materials.append(mat_red)
        else:
            data.objects['Sphere.' + str(i).zfill(3)].data.materials.append(mat_red)


## Return to object mode..
ops.object.mode_set(mode='OBJECT')

## Camera Settings
ops.object.camera_add(enter_editmode=False, align='VIEW')
cam_1 = context.scene.objects['Camera']
cam_1.data.lens = 10
cam_1.data.sensor_width = 7.2481
cam_1.data.clip_start = 0.01
cam_1.data.shift_x = 0
cam_1.data.shift_y = 0
cam_1.location = (0, 0, 0)
cam_1.rotation_euler = Euler((0.0 * pi / 180, 180.0 * pi / 180, 180 * pi / 180))

ops.object.camera_add(enter_editmode=False, align='VIEW')
cam_2 = context.scene.objects['Camera.001']
cam_2.data.lens = 10
cam_2.data.sensor_width = 7.2481
cam_2.data.clip_start = 0.01
cam_2.data.shift_x = 0
cam_2.data.shift_y = 0
cam_2.location = (1, 0, 0)
cam_2.rotation_euler = Euler((0.0 * pi / 180, 90.0 * pi / 180, 0 * pi / 180))

ops.object.camera_add(enter_editmode=False, align='VIEW')
cam_3 = context.scene.objects['Camera.002']
cam_3.data.lens = 10
cam_3.data.sensor_width = 7.2481
cam_3.data.clip_start = 0.1
cam_3.data.shift_x = 0
cam_3.data.shift_y = 0
cam_3.location = (0, -0.2, -0.2)
cam_3.rotation_euler = Euler((0.0 * pi / 180, 150.0 * pi / 180, 270 * pi / 180))

## Light Settings
bpy.ops.object.light_add(type='POINT', align='WORLD', location=(0, 0, 0))
light = context.scene.objects['Point']
light.location = (0, 0, 0)
light.rotation_euler = Euler((90.0 * pi / 180, 0.0 * pi / 180, 90 * pi / 180))
light.data.energy = 40

context.scene.render.image_settings.file_format = 'PNG'
context.scene.render.resolution_x = 640
context.scene.render.resolution_y = 480

if int(args.transparent_mode) == 1:
    context.scene.render.film_transparent = True

## Render
if int(args.viewpoint_mode) == 1:
    context.scene.camera = cam_1
    context.scene.render.filepath = args.save_path
    ops.render.render(use_viewport = True, write_still = True)

elif int(args.viewpoint_mode) == 2:
    context.scene.camera = cam_2
    context.scene.render.filepath = args.save_path
    ops.render.render(use_viewport = True, write_still = True)

elif int(args.viewpoint_mode) == 3:
    context.scene.camera = cam_3
    context.scene.render.filepath = args.save_path
    ops.render.render(use_viewport = True, write_still = True)

else:
    print('[ERROR] viewpoint_mode invalid.')
    exit()
