from action import go_A, go_A_with_B, open_A, close_A, pick_A, place_A_on_B, place_A_in_B, toggle_on_A, toggle_off_A, wait_N, cook_A, cut_A_with_B, add_to_A_with_B
from video_recorder import video_recorder

import argparse
import time
import os
import numpy as np
from math import atan2
from transforms3d.euler import euler2quat

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson import object_states
from omnigibson.robots import Fetch
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.asset_utils import get_og_avg_category_specs, get_og_scene_path
from omnigibson.maps import SegmentationMap, TraversableMap
from omnigibson.utils.control_utils import IKSolver
import omnigibson.utils.transform_utils as T
from omnigibson.sensors import VisionSensor

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import yaml

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.USE_GPU_DYNAMICS = True
    
def cook_chicken(scene_model, headless=False):
    """
    This function enables the robot to cook the chciken in the electric_refrigerator on cookie_sheet using o oven
    """

    trav_map_img = Image.open(os.path.join(get_og_scene_path(
        scene_model), "layout", "floor_trav_0.png"))
    # test traversable map
    trav_map_original_size = np.array(trav_map_img).shape[0]
    trav_map_size = int(trav_map_original_size * 0.1)
    trav_map_erosion = 2
    trav_map_img = np.array(trav_map_img.resize(
        (trav_map_size, trav_map_size)))
    trav_map_img = cv2.erode(trav_map_img, np.ones(
        (trav_map_erosion, trav_map_erosion)))
    
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
    knife = og.sim.scene.object_registry("name", "carving_knife_198")
    board = og.sim.scene.object_registry("name", "chopping_board_189")
    stockpot = og.sim.scene.object_registry("name", "stockpot_190")
    cabinet = og.sim.scene.object_registry("name", "bottom_cabinet_no_top_qudfwe_0")
    chicken = og.sim.scene.object_registry("name", "chicken_188")
    fridge = og.sim.scene.object_registry("name", "fridge_dszchb_0")
    oven = og.sim.scene.object_registry("name", "oven_wuinhm_0")
    salt_shaker = og.sim.scene.object_registry("name", "salt_shaker_197")
    salt = og.sim.scene.system_registry("name", "salt")
    table = og.sim.scene.object_registry("name", "breakfast_table_skczfi_1")

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/cook_chicken", 
                       name="cook_chicken", trav_map_img=trav_map_img)

    # go_A(fridge, robot, trav_map, trav_map_size)
    # open_A(fridge, robot)
    # pick_A(chicken, robot)
    # close_A(fridge, robot)
    # go_A_with_B(board, chicken, robot, trav_map, trav_map_size)
    # place_A_on_B(chicken, board, robot)
    # go_A(cabinet, robot, trav_map, trav_map_size)
    # open_A(cabinet, robot)
    # pick_A(knife, robot)
    # go_A_with_B(board, knife, robot, trav_map, trav_map_size)
    # cut_A_with_B(chicken, knife, robot)
    # diced_chicken = og.sim.system_registry("name", "diced__chicken")
    # go_A(cabinet, robot, trav_map, trav_map_size)
    # pick_A(stockpot, robot)
    # go_A_with_B(oven, stockpot, robot, trav_map, trav_map_size)
    # open_A(oven, robot)
    # place_A_in_B(stockpot, oven, robot)
    # go_A(board, robot, trav_map, trav_map_size)
    # pick_A(diced_chicken, robot)
    # go_A_with_B(oven, diced_chicken, robot, trav_map, trav_map_size)
    # place_A_in_B(diced_chicken, stockpot, robot)
    # go_A(cabinet, robot, trav_map, trav_map_size)
    # pick_A(salt_shaker, robot)
    # close_A(cabinet, robot)
    # go_A_with_B(board, salt_shaker, robot, trav_map, trav_map_size)
    # add_to_A_with_B(stockpot, salt_shaker, robot)
    # close_A(oven, robot)
    # toggle_on_A(oven, robot)
    # cook_A(diced_chicken, robot)
    # toggle_off_A(oven, robot)
    # print(stockpot.states[object_states.Contains].get_value(cooked_diced_chicken))
    # print(stockpot.states[object_states.Contains].get_value(salt))

    go_A_with_B(oven, salt_shaker, robot, trav_map, trav_map_size)
    open_A(oven, robot)
    place_A_in_B(stockpot, oven, robot)
    add_to_A_with_B(stockpot, salt_shaker, robot)
    print(stockpot.states[object_states.Contains].get_value(salt))

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    """
    This demo shows loading behavior task "cook_eggplant"
    And record a video about how to complete the task
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Load the pre-selected cpnfiguration
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_0_int"
    cfg["task"]["activity_name"] = "cook_chicken"
    cfg["scene"]["not_load_object_categories"] = ["ceils"]

    # Load the environment
    env = og.Environment(configs=cfg)
    
    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([2.32248, -8.74338, 9.85436]),
        orientation=np.array([0.39592, 0.13485, 0.29286, 0.85982]),
    )

    cook_chicken(cfg["scene"]["scene_model"])

    # # Loop until the user requests an exit
    # exit_now = False
    # while not exit_now:
    #     og.sim.step()

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
