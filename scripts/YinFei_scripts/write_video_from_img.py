import os
from moviepy.editor import *
import path_settings

 
img_file_paths = []

img_offset = 0
img_dir = '/home/inffzy/nutstore_files/arclab_research/ARCLab-CCCatheter/results/UN007/D00_0007/images' 


for i in range(0, 21):
    img_file_paths.append(os.path.join(img_dir, str(i + img_offset).zfill(3) + '.png'))

clip = ImageSequenceClip(img_file_paths, fps=4)
clip.write_videofile(os.path.join(path_settings.video_save_dir, 'UN007_D00_0007.mp4'), fps=30)

#clip.write_gif('')