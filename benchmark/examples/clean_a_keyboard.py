from action import go_A, open_A, pick_A, go_A_with_B, place_A_at_P, toggle_on_A, toggle_off_A, close_A, uncover_A_with_B
from video_recorder import video_recorder

import argparse
import time
import os
import numpy as np
from math import atan2
from transforms3d.euler import euler2quat

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson import object_states
from omnigibson.robots import Fetch
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.asset_utils import get_og_avg_category_specs, get_og_scene_path
from omnigibson_modified import TraversableMap
from omnigibson.utils.control_utils import IKSolver
import omnigibson.utils.transform_utils as T
from omnigibson.sensors import VisionSensor

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import yaml

from utils import (
    get_trav_map,
    convert
)

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.USE_GPU_DYNAMICS = True
    
def clean_a_keyboard(scene_model, headless=False):
    """
    This function enables the robot to cook the eggplant in the electric_refrigerator on cookie_sheet using o oven
    """

    trav_map_img, trav_map_size = get_trav_map(scene_model)
    trav_map = TraversableMap()
    trav_map.load_map(os.path.join(get_og_scene_path(
            scene_model), "layout"))
    trav_map.build_trav_graph(trav_map_size, os.path.join(get_og_scene_path(
            scene_model), "graph"), 1, trav_map_img.copy())
    
    # Create figure
    if not headless:
        plt.figure(figsize=(12, 12))
        plt.imshow(trav_map_img)
        plt.title(f"Traversable area of {scene_model} scene")

    # if not headless:
    #     plt.show()

    # don't know why, but double loading is needed
    trav_map = TraversableMap()
    trav_map.load_map(os.path.join(get_og_scene_path(
        scene_model), "layout"))
    trav_map.build_trav_graph(trav_map_size, os.path.join(get_og_scene_path(
        scene_model), "graph"), 1, trav_map_img)
    
    # Capture the reference of objects
    robot = og.sim.scene.object_registry("name", "robot0")
    keyboard = og.sim.scene.object_registry("name", "keyboard_188")
    sink = og.sim.scene.object_registry("name", "sink_zexzrc_0")
    stain = og.sim.scene.system_registry("name", "stain")

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/clean_a_keyboard", \
        name="clean_a_keyboard", trav_map_img=trav_map_img)

    # # Lift the arm
    # move_arm_to_target(get_arm_position(robot), robot)
    # for _ in range(300):
    #             og.sim.step()

    go_A(keyboard, robot, trav_map, trav_map_size)
    pick_A(keyboard, robot)
    go_A_with_B(sink, keyboard, robot, trav_map, trav_map_size)
    place_A_at_P(keyboard, sink.states[object_states.ParticleSource].link.get_position() + np.array([0, 0, -0.2]), robot)
    toggle_on_A(sink, robot)
    uncover_A_with_B(keyboard, stain, robot)
    toggle_off_A(sink, robot)
    pick_A(keyboard, robot)

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    """
    This demo shows loading behavior task "clean_a_keyboard"
    And record a video about how to complete the task
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Load the pre-selected cpnfiguration
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_0_int"
    cfg["task"]["activity_name"] = "clean_a_keyboard"
    cfg["scene"]["not_load_object_categories"] = ["ceils"]

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    clean_a_keyboard(cfg["scene"]["scene_model"])

    # # Loop until the user requests an exit
    # exit_now = False
    # while not exit_now:
    #     og.sim.step()

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
