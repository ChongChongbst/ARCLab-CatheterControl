import numpy as np

import path_settings
import os
import sys
sys.path.append(path_settings.scripts_dir)

import YinFei_scripts.camera_settings as camera_settings
from YinFei_scripts.camera_settings import SimulationExperiment
from YinFei_scripts.data_deneration import DataGeneration
from YinFei_scripts.experiment_setup import experiments