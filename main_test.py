#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
import time 
import os
import sys
import numpy as np
import json

import platform 

from config import Config

from slam import Slam, SlamState
from slam_plot_drawer import SlamPlotDrawer
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import DatasetType, SensorType
from trajectory_writer import TrajectoryWriter

from viewer3D import Viewer3D
from utils_sys import getchar, Printer, force_kill_all_and_exit
from utils_img import ImgWriter
from utils_eval import eval_ate
from utils_geom_trajectory import find_poses_associations
from utils_colors import GlColors
from utils_serialization import SerializableEnumEncoder

from feature_tracker_configs import FeatureTrackerConfigs

from loop_detector_configs import LoopDetectorConfigs

from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
from utils_depth import img_from_depth, filter_shadow_points

from frame import Frame

from config_parameters import Parameters  

from rerun_interface import Rerun

from datetime import datetime
import traceback

import argparse

import inspect
import g2o


datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
def draw_associated_cameras(viewer3D, assoc_est_poses, assoc_gt_poses, T_gt_est):       
    T_est_gt = np.linalg.inv(T_gt_est)
    scale = np.mean([np.linalg.norm(T_est_gt[i, :3]) for i in range(3)])
    R_est_gt = T_est_gt[:3, :3]/scale # we need a pure rotation to avoid camera scale changes
    assoc_gt_poses_aligned = [np.eye(4) for i in range(len(assoc_gt_poses))]
    for i in range(len(assoc_gt_poses)):
        assoc_gt_poses_aligned[i][:3,3] = T_est_gt[:3, :3] @ assoc_gt_poses[i][:3, 3] + T_est_gt[:3, 3]
        assoc_gt_poses_aligned[i][:3,:3] = R_est_gt @ assoc_gt_poses[i][:3,:3]
    viewer3D.draw_cameras([assoc_est_poses, assoc_gt_poses_aligned], [GlColors.kCyan, GlColors.kMagenta])    

def list_signatures(cls):
    for name, member in inspect.getmembers(cls):
        print(name, member)
        if inspect.isfunction(member):
            signature = inspect.signature(member)
            print(f"{name}{signature}")

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Optional path for custom configuration file')
    parser.add_argument('--no_output_date', action='store_true', help='Do not append date to output directory')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')    
    args = parser.parse_args()
    
    if args.config_path:
        config = Config(args.config_path) # use the custom configuration path file
    else:
        config = Config()

    # list_signatures(g2o.Isometry3d)
    list_signatures(Slam)

    