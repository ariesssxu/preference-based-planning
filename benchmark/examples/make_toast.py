import os
import numpy as np
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import choose_from_options
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from omnigibson.utils.asset_utils import get_available_og_scenes, get_og_scene_path
from omnigibson_modified import TraversableMap
from utils import get_trav_map
import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.objects import DatasetObject
from action import go_A, go_A_with_B, open_A, close_A, pick_A, place_A_on_B, place_A_in_B, toggle_on_A, toggle_off_A, wait_N, cook_A, cut_A_with_B, add_to_A_with_B, heat_A, close_A_with_B, open_A_with_B, place_A_at_P, place_A_at_P_rotate_O
from omnigibson.sensors import VisionSensor
import cv2
from video_recorder import video_recorder
from omnigibson.systems.system_base import get_system, is_system_active, is_visual_particle_system, \
    is_physical_particle_system, SYSTEM_REGISTRY, add_callback_on_system_init, add_callback_on_system_clear, \
    remove_callback_on_system_init, remove_callback_on_system_clear, import_og_systems
from omnigibson.systems.micro_particle_system import *
from omnigibson.systems.macro_particle_system import *


# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = True

def make_toast(scene_model, headless=False):
    """
    This function enables the robot to heat food from the electric_refrigerator in the microwave
    (:goal 
        (and 
            (forall 
                (?bread_slice.n.01 - bread_slice.n.01)
                (cooked ?bread_slice.n.01)
            )
        )
    )
    """
    trav_map_img = Image.open(os.path.join(get_og_scene_path(scene_model), "layout", "floor_trav_0.png"))
    # test traversable map
    trav_map_original_size = np.array(trav_map_img).shape[0]
    trav_map_size = int(trav_map_original_size * 0.1)
    trav_map_erosion = 2
    trav_map_img = np.array(trav_map_img.resize((trav_map_size, trav_map_size)))
    trav_map_img = cv2.erode(trav_map_img, np.ones((trav_map_erosion, trav_map_erosion)))

    # Create figure
    if not headless:
        plt.figure(figsize=(12, 12))
        plt.imshow(trav_map_img)
        plt.title(f"Traversable area of {scene_model} scene")

    # make sure the graph folder exists
    og_scene_path = os.path.join(get_og_scene_path(scene_model), "graph")
    if not os.path.exists(og_scene_path):
        os.makedirs(og_scene_path)
    
    # don't know why, but double loading is needed
    trav_map = TraversableMap()
    trav_map.load_map(os.path.join(get_og_scene_path(scene_model), "layout"))
    trav_map.build_trav_graph(trav_map_size, os.path.join(get_og_scene_path(
        scene_model), "graph"), 1, trav_map_img)
    
    # Capture the reference of objects 
    robot = og.sim.scene.object_registry("name", "robot0")
    countertop = og.sim.scene.object_registry("name", "countertop_tpuwys_2")
    bread_slice_1 = og.sim.scene.object_registry("name", "bread_slice_86")
    bread_slice_2 = og.sim.scene.object_registry("name", "bread_slice_87")
    cabinet = og.sim.scene.object_registry("name", "bottom_cabinet_no_top_vzzafs_0")
    toaster = og.sim.scene.object_registry("name", "toaster_85")

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/make_toast", name="make_toast", trav_map_img=trav_map_img)
    
    print(">>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>")
    print("1. take bread_slice_1 from cabinet to toaster")
    # 1. take bread_slice_1 from cabinet to toaster
    go_A(cabinet, robot, trav_map, trav_map_size)
    open_A(cabinet, robot)
    pick_A(bread_slice_1, robot)
    # close_A_with_B(cabinet, bread_slice_1, robot)
    print("open cabinet: ", cabinet.states[object_states.Open].get_value())

    go_A_with_B(toaster, bread_slice_1, robot, trav_map, trav_map_size)
    place_A_on_B(bread_slice_1, toaster, robot)
    
    # 2. cook bread_slice_1
    toggle_on_A(toaster, robot)
    print("toggle on ")
    cook_A(bread_slice_1, robot)
    toggle_off_A(toaster, robot)
    print("toggle off ")

    # 3. take bread_slice_1 from toaster to countertop
    pick_A(bread_slice_1, robot)
    place_A_on_B(bread_slice_1, countertop, robot)

    print("cooked bread_slice_1: ", bread_slice_1.states[object_states.Cooked].get_value())

    print(">>>>>>>>>>>>>>>>>>>")
    go_A(cabinet, robot, trav_map, trav_map_size)
    # open_A(cabinet, robot)
    pick_A(bread_slice_2, robot)
    close_A_with_B(cabinet, bread_slice_2, robot)

    go_A_with_B(toaster, bread_slice_2, robot, trav_map, trav_map_size)
    place_A_on_B(bread_slice_2, toaster, robot)
    
    toggle_on_A(toaster, robot)
    print("toggle on ")
    cook_A(bread_slice_2, robot)
    toggle_off_A(toaster, robot)
    print("toggle off ")

    pick_A(bread_slice_2, robot)
    place_A_at_P(bread_slice_2, countertop.get_position() + np.array([bread_slice_2.native_bbox[0], 0.0, bread_slice_2.native_bbox[2] * 0.5]), robot)

    print("cooked bread_slice_2: ", bread_slice_2.states[object_states.Cooked].get_value())

    for _ in range(20):
        video_recorder.get_video()
        og.sim.step()
    
    # Release the video
    video_recorder.release()


def main(random_selection=False, headless=False, short_exec=False):
    """
    This demo shows loading behavior task "cook_beef"
    And record a video about how to complete the task

    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Benevolence_1_int"
    cfg["task"]["activity_name"] = "make_toast"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.2125454 ,  0.61283677,  2.13839736]),   
        orientation=np.array([-0.00445776,  0.24650814,  0.96897205, -0.01752249]),
    )
   
    for obj in env.scene.objects:
        if any([name in obj.name for name in ["floor", "wall", "ceiling"]]):
            continue
        print(obj.name,"   ", type(obj))

    make_toast(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()


if __name__ == "__main__":
    main()