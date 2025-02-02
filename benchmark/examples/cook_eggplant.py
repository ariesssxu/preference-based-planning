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
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from action import go_A, go_A_with_B, open_A, pick_A, close_A, place_A_on_B, cut_A_with_B, toggle_on_A, toggle_off_A, cook_A
from video_recorder import video_recorder
import yaml

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True
gm.USE_GPU_DYNAMICS = True
    
def cook_eggplant(scene_model, headless=False):
    """
    This function enables the robot to cook the eggplant in the electric_refrigerator on cookie_sheet using o oven
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
    knife = og.sim.scene.object_registry("name", "carving_knife_190")
    board = og.sim.scene.object_registry("name", "chopping_board_191")
    sheet = og.sim.scene.object_registry("name", "baking_sheet_188")
    countertop = og.sim.scene.object_registry("name", "countertop_tpuwys_2")
    eggplant = og.sim.scene.object_registry("name", "eggplant_189")
    fridge = og.sim.scene.object_registry("name", "fridge_dszchb_0")
    oven = og.sim.scene.object_registry("name", "oven_wuinhm_0")

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 12.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, \
        save_path=f"{og.root_path}/../../images/cook_eggplant", name="cook_eggplant", trav_map_img=trav_map_img)

    # # Lift the arm
    # move_arm_to_target(get_arm_position(robot), robot)
    # for _ in range(300):
    #             og.sim.step()

    # test cut
    apple = DatasetObject(
            name="apple",
            category="apple",
            model="agveuv",
            # bounding_box=get_og_avg_category_specs().get("apple"),
            fit_avg_dim_volume=True,
            # position=[0, 0, 50.0],
            position=[-6.848034381866455,
                        -1.78300940990448,
                        0.8840968608856201]
    )
    og.sim.import_object(apple)
    # Settle the apple
    for _ in range(100):
        og.sim.step(np.array([]))

    # # Pick the eggplant in the electric refrigerator
    # go_A(fridge, robot, trav_map, trav_map_size, video_writer)
    # open_A(fridge, robot, video_writer)
    # pick_A(eggplant, robot, video_writer)
    # close_A(fridge, robot, video_writer)
    
    # # Place the eggplant on the chooping board
    # go_A_with_B(board, eggplant, robot, trav_map, trav_map_size, video_writer)
    # place_A_on_B(eggplant, board, robot, video_writer)

    # # Pick the carving knife on the chooping board
    # pick_A(knife, robot, video_writer)
    # # Cut the eggplant into half
    # cut_A(eggplant, knife, robot, video_writer)
                
    # # Pick and the baking sheet on the countertop
    # go_A(sheet, robot, trav_map, trav_map_size, video_writer)
    # pick_A(sheet, robot, video_writer)
    # # Place the baking sheet on the oven
    # go_A_with_B(oven, sheet, robot, trav_map, trav_map_size, video_writer)
    # place_A_on_B(sheet, oven, robot, video_writer)

    # # Pick and place the half eggplant on the baking sheet
    # go_A(board, robot, trav_map, trav_map_size, video_writer)
    # pick_A(half_eggplant, robot, video_writer)
    # go_A_with_B(oven, half_eggplant, robot, trav_map, trav_map_size, video_writer)
    # place_A_on_B(half_eggplant, sheet, robot, video_writer)
    
    # # Operate oven to cook eggplant
    # toggle_on_A(oven, robot, video_writer)
    # cook_A(eggplant, robot, video_writer)
    # toggle_off_A(oven, robot, video_writer)
    
    # Place the eggplant on the chooping board
    go_A_with_B(board, apple, robot, trav_map, trav_map_size)
    place_A_on_B(apple, board, robot)

    # Pick the carving knife on the chooping board
    pick_A(knife, robot)
    # Cut the eggplant into half
    cut_A_with_B(apple, knife, robot)
    
    half_eggplant = og.sim.scene.object_registry("name", "half_apple_0")
                
    # Pick and the baking sheet on the countertop
    go_A(sheet, robot, trav_map, trav_map_size)
    pick_A(sheet, robot)
    # Place the baking sheet on the oven
    go_A_with_B(oven, sheet, robot, trav_map, trav_map_size)
    place_A_on_B(sheet, oven, robot)

    # Pick and place the half eggplant on the baking sheet
    go_A(board, robot, trav_map, trav_map_size)
    pick_A(half_eggplant, robot)
    go_A_with_B(oven, half_eggplant, robot, trav_map, trav_map_size)
    place_A_on_B(half_eggplant, sheet, robot)
    
    # Operate oven to cook eggplant
    toggle_on_A(oven, robot)
    cook_A(half_eggplant, robot)
    toggle_off_A(oven, robot)

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
    cfg["task"]["activity_name"] = "cook_eggplant"

    # Load the environment
    env = og.Environment(configs=cfg)
    
    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([2.32248, -8.74338, 9.85436]),
        orientation=np.array([0.39592, 0.13485, 0.29286, 0.85982]),
    )

    cook_eggplant(cfg["scene"]["scene_model"])

    # # Loop until the user requests an exit
    # exit_now = False
    # while not exit_now:
    #     og.sim.step()

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
