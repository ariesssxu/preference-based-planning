import argparse
import time
import os
import numpy as np
from math import atan2
from transforms3d.euler import euler2quat
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson import object_states
from omnigibson.robots import Fetch
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.utils.asset_utils import get_available_og_scenes, get_og_scene_path
from omnigibson_modified import TraversableMap
from omnigibson.utils.control_utils import IKSolver
from omnigibson.utils.asset_utils import (
    get_og_avg_category_specs,
)
import yaml
from omnigibson.sensors import VisionSensor
from omnigibson.simulator import launch_simulator
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
from utils import (
    get_trav_map,
    convert, 
    robot_rotate_interpolation,
)
from action import *
from video_recorder import video_recorder
from draw_figs import terminate, init_keyboard_event_handler

# logging.disable(logging.ERROR)
# logging.disable(logging.WARNING)

# in 2023.1: call launch_simulator() to start the simulator
launch_simulator()

def pick_and_place(obj, pre_recep, is_on_floor, is_in, target_recep, target_pos, will_on_floor, robot, scene_model, \
                   trav_map=None, trav_map_img=None, trav_map_size=None, headless=False):
    """
    This function enables the robot to pick a object and place it to a target position
    obj: the object to pick [an omnigibson object]
    pre_recep: previous receptacle for the object to pick [an omnigibson object or None]
    is_on_floor: whether the object is on the floor, True for is and False for isn't [a bool]
    is_in: whether the object is in a receptacle, True for is and False for isn.t [a bool]
    target_recep: target receptacle for the object to place [an omnigibson object or None]
    target_pos: targe position for the object to place [a 3-array]
    will_on_floor: whether the object will be place on the floor, True for will and False for won't [a bool]
    robot: [fetch robot of omnigibson]
    scene_model: the name of the loaded scene model [a string]
    """

    paths = []
    
    if not trav_map_size:
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
    
    # Lift the arm
    # move_arm_to_target(get_arm_position(robot), robot)
    # terminate()
    for _ in range(10):
        og.sim.step()

    # Find and pick the object
    if is_on_floor:
        paths.append(go_A(obj, robot, trav_map, trav_map_size))
    else:
        paths.append(go_A(pre_recep, robot, trav_map, trav_map_size))
    if is_in:
        open_A(pre_recep, robot)
        pick_A(obj, robot)
        close_A(pre_recep, robot)
        robot.reset()
        # terminate()
    else:
        pick_A(obj, robot)
    
    # Find the target place and place the object
    if will_on_floor:
        paths.append(go_P_with_B(target_pos, obj, robot, trav_map, trav_map_size))
        place_A_at_P(obj, target_pos, robot)
    else:
        paths.append(go_A_with_B(target_recep, obj, robot, trav_map, trav_map_size))
        if "openable" in target_recep.abilities:
            open_A(target_recep, robot)
            place_A_at_P(obj, target_pos, robot)
            close_A(target_recep, robot)
        else:
            place_A_at_P(obj, target_pos, robot)

    return paths

def main(random_selection=False, headless=False, short_exec=False):
    """
    This demo shows loading an apple and a fetch robot in an interactive scene
    This will sample the position of the apple and the robot
    The robot will go to the position of the apple and pick it
    And then go to the position of the table and put the apple on it
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests and we
    # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    if not (random_selection and headless and short_exec):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--programmatic",
            "-p",
            dest="programmatic_pos",
            action="store_true",
            help="if the IK solvers should be used with the GUI or programmatically",
        )
        args = parser.parse_args()
        programmatic_pos = args.programmatic_pos
    else:
        programmatic_pos = True

    # Load an interactive scene "Rs_int"
    # scene_model = "Beechwood_0_int"
    scene_model = "Rs_int"

    trav_map_img, trav_map_size = get_trav_map(scene_model)
    trav_map = TraversableMap()
    trav_map.load_map(os.path.join(get_og_scene_path(
            scene_model), "layout"))
    trav_map.build_trav_graph(trav_map_size, os.path.join(get_og_scene_path(
            scene_model), "layout"), 1, trav_map_img.copy())
    
    self_path = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(self_path, "./draw_figs/scene_config.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = scene_model

    env = og.Environment(configs=cfg)

    robot = env.robots[0]

    # Update the viewer camera's pose sothat it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([2.32248, -8.74338, 9.85436]),
        orientation=np.array([0.39592, 0.13485, 0.29286, 0.85982]),
    )

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            print(sensor.horizontal_aperture)
            sensor.horizontal_aperture = 50
    
    # At least one simulation step while the simulator is playing must occur for the robot (or in general, any object)
    # to be fully initialized after it is imported into the simulator
    # robot.set_position([-2.6, 3.0, 0.01])
    og.sim.play()
    for i in range(100):
        og.sim.step()
    # Make sure none of the joints are moving
    robot.keep_still()

    terminate()
    
    # in 2023.1: call without boundingbox
    apple = DatasetObject(
            name="apple",
            category="apple",
            model="agveuv",
            # bounding_box=avg_category_spec.get("apple"),
            fit_avg_dim_volume=True,
            # position=[0, 0, 50.0],
            position=[-5.848034381866455,
                        -1.78300940990448,
                        0.7840968608856201],
            abilities={"diceable": {}}
    )
    og.sim.import_object(apple)
    # Place the apple so it rests on the floor
    center_offset = apple.get_position() - apple.aabb_center + np.array([0, 0, apple.aabb_extent[2] / 2.0])
    center_offset[0] += 0.8
    center_offset[1] += 1.8
    apple.set_position(center_offset)
    # Settle the apple
    for _ in range(100):
        og.sim.step(np.array([]))
    apple.set_orientation([0.707, 0, 0, 0.707])
    # Settle the apple
    for _ in range(100):
        og.sim.step(np.array([]))

    # Capture the reference of the fridge
    fridge = og.sim.scene.object_registry("name", "fridge_dszchb_0")
    # fridge = og.sim.scene.object_registry("name", "fridge_xyejdx_0")
    apple.set_position(fridge.get_position())
    # Settle the apple
    for _ in range(100):
        og.sim.step(np.array([]))

    # Capture the reference of the countertop
    countertop = og.sim.scene.object_registry("name", "breakfast_table_skczfi_0")
    # countertop = og.sim.scene.object_registry("name", "breakfast_table_skczfi_1")
    target_pos = countertop.get_position()
    target_pos[2] += 0.5 * apple.native_bbox[2] + 0.5 * countertop.native_bbox[2]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Pick and Place
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/pick_and_place_whole", \
        name="pick_and_place",trav_map_img=trav_map_img)
    pick_and_place(apple, fridge, False, True, countertop, target_pos, False, robot, scene_model, trav_map, trav_map_img, trav_map_size)
    video_recorder.release()

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
