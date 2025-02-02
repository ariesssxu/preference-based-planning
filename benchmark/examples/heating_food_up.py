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
from action import go_A, go_A_with_B, open_A, close_A, pick_A, place_A_on_B, place_A_in_B, toggle_on_A, toggle_off_A, wait_N, cook_A, cut_A_with_B, add_to_A_with_B, heat_A, close_A_with_B, open_A_with_B, place_A_at_P
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

def heating_food_up(scene_model, headless=False):
    """
    This function enables the robot to heat food from the electric_refrigerator in the microwave
    (:goal 
        (and  
            (hot ?hamburger.n.01_1)
            (ontop ?hamburger.n.01_1 ?countertop.n.01_1)
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
    hamburger = og.sim.scene.object_registry("name", "hamburger_85")
    plate = og.sim.scene.object_registry("name", "plate_86")
    fridge = og.sim.scene.object_registry("name", "fridge_xyejdx_0")
    countertop = og.sim.scene.object_registry("name", "countertop_tpuwys_4")
    microwave = og.sim.scene.object_registry("name", "microwave_bfbeeb_0")

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/heating_food_up", name="heating_food_up", trav_map_img=trav_map_img)
    
    print(">>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>")
    print("1. take hamburger from fridge to microwave")
    # 1. take hamburger from fridge to microwave
    go_A(fridge, robot, trav_map, trav_map_size)
    open_A(fridge, robot)
    pick_A(hamburger, robot)
    close_A_with_B(fridge, hamburger, robot)

    go_A_with_B(microwave, hamburger, robot, trav_map, trav_map_size)
    open_A_with_B(microwave, hamburger, robot)
    place_A_in_B(hamburger, microwave, robot)
    close_A(microwave, robot)
    
    print(">>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>")
    print("2. heating hamburger")
    # 2. heating hamburger
    print(">>>>>>>>>>>>>>>>>>>")
    print("Start heating ...")
    toggle_on_A(microwave, robot)
    heat_A(hamburger, robot)
    toggle_off_A(microwave, robot)


    print(">>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>")
    print("3. put hamburger on countertop")
    # 3. put hamburger on countertop
    open_A(microwave, robot)
    pick_A(hamburger, robot)
    close_A_with_B(microwave, hamburger, robot)
    go_A_with_B(countertop, hamburger, robot, trav_map, trav_map_size)
    place_A_on_B(hamburger, countertop, robot)

    for _ in range(20):
        video_recorder.get_video()
        og.sim.step()
    
    print("Hambuger frozen state: ", hamburger.states[object_states.Frozen].get_value())
    print("Hambuger heated state: ", hamburger.states[object_states.Heated].get_value())
    print("Ontop plate: ", hamburger.states[object_states.OnTop].get_value(plate))
    print("Inside plate: ", hamburger.states[object_states.Inside].get_value(plate))
    
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
    cfg["task"]["activity_name"] = "heating_food_up"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.87162246, 0.21676164, 2.05417262]),
        orientation=np.array([0.43026783, 0.26601513, 0.453621  , 0.73371216]),
    )
   
    for obj in env.scene.objects:
        if any([name in obj.name for name in ["floor", "wall", "ceiling"]]):
            continue
        print(obj.name,"   ", type(obj))

    heating_food_up(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()


if __name__ == "__main__":
    main()