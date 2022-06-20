import numpy as np

focal_length = 10

image_size_x = 640  ## px
image_size_y = 480  ## px
aspect_ratio = image_size_x / image_size_y

sensor_size_x = 7.248  ## (mm) sensor_width in Blender
sensor_size_y = sensor_size_x / aspect_ratio

a = focal_length / sensor_size_x * image_size_x
b = focal_length / sensor_size_y * image_size_y

center_x = 320
center_y = 240

intrinsics = np.array([
    [a, 0, center_x],
    [0, b, center_y],
    [0, 0, 1       ]])
print('Camera Intrinsics:')
print(intrinsics)


location = np.array([0, 0, 0]).reshape((3, 1))
rotation_euler = np.array([0, np.pi, np.pi])

phi = rotation_euler[0]
theta = rotation_euler[1]
psi = rotation_euler[2]

rotation_matrix = np.array([
    [np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
     np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi),
     np.sin(psi) * np.sin(theta)], 
    [-1 * np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi),
     -1 * np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi),
     np.cos(psi) * np.sin(theta)], 
    [np.sin(theta) * np.sin(phi),
     -1 * np.sin(theta) * np.cos(phi),
     np.cos(theta)]])


extrinsics = np.concatenate((rotation_matrix, location), axis=1)
extrinsics = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)
print('Camera Extrinsics:')
print(extrinsics)

