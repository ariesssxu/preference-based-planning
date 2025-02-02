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
from action import go_A, go_A_with_B, open_A, close_A, pick_A, place_A_on_B, place_A_in_B, toggle_on_A, toggle_off_A, wait_N, cook_A, cut_A_with_B, add_to_A_with_B, heat_A, close_A_with_B, open_A_with_B, place_A_at_P, uncover_A_with_B
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
    
def packing_art_supplies_into_car(scene_model, headless=False):
    """
    (:goal 
        (and 
            (forall 
                (?pad.n.01 - pad.n.01) 
                (inside ?pad.n.01 ?ashcan.n.01_1)
            ) 
            (forall 
                (?can__of__soda.n.01 - can__of__soda.n.01) 
                (inside ?can__of__soda.n.01 ?ashcan.n.01_1)
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
    trav_map.build_trav_graph(trav_map_size, os.path.join(get_og_scene_path(scene_model), "graph"), 1, trav_map_img)
    
    
    # Capture the reference of objects
    robot = og.sim.scene.object_registry("name", "robot0")
    bag = og.sim.scene.object_registry("name", "suitcase_276")
    car = og.sim.scene.object_registry("name", "car_275")
    driveway = og.sim.scene.object_registry("name", "driveway_aipswu_0")
    floor = og.sim.scene.object_registry("name", "floors_cyanxu_0")
    marker_1 = og.sim.scene.object_registry("name", "marker_278")
    marker_2 = og.sim.scene.object_registry("name", "marker_279")
    pencil = og.sim.scene.object_registry("name", "colored_pencil_277")

    # og.sim.remove_object(marker_2)

    og.sim.step()

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/packing_art_supplies_into_car", name="packing_art_supplies_into_car", trav_map_img=trav_map_img)

    # go_A(can_of_soda_1, robot, trav_map, trav_map_size)
    # pick_A(can_of_soda_1, robot)
    # go_A_with_B(sink, can_of_soda_1, robot, trav_map, trav_map_size)
    # place_A_in_B(can_of_soda_1, trash_can, robot)
    # print("inside can_of_soda_1: ", trash_can.states[object_states.Inside].get_value(can_of_soda_1))

    # go_A(can_of_soda_2, robot, trav_map, trav_map_size)
    # pick_A(can_of_soda_2, robot)
    # go_A_with_B(sink, can_of_soda_2, robot, trav_map, trav_map_size)
    # place_A_in_B(can_of_soda_2, trash_can, robot)
    # print("inside can_of_soda_2: ", trash_can.states[object_states.Inside].get_value(can_of_soda_2))

    # go_A(shelf, robot, trav_map, trav_map_size)
    # pick_A(pad_1, robot)
    # go_A_with_B(sink, pad_1, robot, trav_map, trav_map_size)
    # place_A_in_B(pad_1, trash_can, robot)
    # print("inside pad_1: ", trash_can.states[object_states.Inside].get_value(pad_1))

    # # go_A(pad_2, robot, trav_map, trav_map_size)
    # # pick_A(pad_2, robot)
    # # go_A_with_B(sink, pad_2, robot, trav_map, trav_map_size)
    # # place_A_in_B(pad_2, trash_can, robot)
    # # print("inside pad_2: ", trash_can.states[object_states.Inside].get_value(pad_2))
    # print("inside can_of_soda_1: ", trash_can.states[object_states.Inside].get_value(can_of_soda_1))
    # print("inside can_of_soda_2: ", trash_can.states[object_states.Inside].get_value(can_of_soda_2))
    # print("inside pad_1: ", trash_can.states[object_states.Inside].get_value(pad_1))


    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Pomaria_0_garden"
    cfg["task"]["activity_name"] = "packing_art_supplies_into_car"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    # og.sim.viewer_camera.set_position_orientation(
    #     position=np.array([-1.97994845,  0.69635405,  1.68618876]),
    #     orientation=np.array([-0.20374329,  0.53802809,  0.76492382, -0.28966532]),
    # )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    packing_art_supplies_into_car(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
