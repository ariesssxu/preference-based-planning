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
from action import *
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

    
def packing_hiking_equipment_into_car(scene_model, headless=False):
    """
        (and 
            (covered ?lawn.n.01_1 ?insectifuge.n.01_1) bush_lfyqkj_0
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

    # door_1 = og.sim.scene.object_registry("name", "door_vudhlc_1")

    # og.sim.remove_object(ceiling_1)
    # og.sim.remove_object(ceiling_2)
    # og.sim.remove_object(wall_1)
    # og.sim.remove_object(wall_2)

    # sleeping_bag.set_position_orientation([9.75466, 5.31473, sleeping_bag.get_position()[2]], sleeping_bag.get_orientation())
    # robot.set_position_orientation([9.41464, 6.89067, robot.get_position()[2]], robot.get_orientation())
    # sleeping_bag.set_position_orientation([-5.93793, 3.43573, sleeping_bag.get_position()[2]], sleeping_bag.get_orientation())
    # robot.set_position_orientation([-3.6236, 3.38747, robot.get_position()[2]], robot.get_orientation())
    # car.set_position_orientation(car.get_position() + np.array([-0.2, 1.0, 0.0]), car.get_orientation())
    # backpack.set_position_orientation(np.array([car.get_position()[0] - 2.0, car.get_position()[1]-1.4, backpack.get_position()[2]]), backpack.get_orientation())

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/packing_hiking_equipment_into_car_P", name="packing_hiking_equipment_into_car_P", trav_map_img=trav_map_img)
    
    # while True:
    #     video_recorder.get_video()
    #     og.sim.step()

    # # 1. pick 
    # pick_A(sleeping_bag, robot)
    # position = np.array([car.get_position()[0], car.get_position()[1]-1.4, 1.17623])
    # go_P_with_B(position, sleeping_bag, robot, trav_map, trav_map_size)
    # place_A_on_B(sleeping_bag, car, robot)

    # for i in range(20):
    #     video_recorder.get_video()
    #     og.sim.step()

    # go_A(backpack, robot, trav_map, trav_map_size)
    # pick_A(backpack, robot)
    # place_A_on_B(backpack, car, robot)

    while True:
        video_recorder.get_video()
        og.sim.step()

    for i in range(50):
        video_recorder.get_video()
        og.sim.step()

    # # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Pomaria_0_garden"
    cfg["task"]["activity_name"] = "packing_hiking_equipment_into_car"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    # og.sim.viewer_camera.set_position_orientation(
    #     position=np.array([-7.8084891 ,  7.75240491, 12.61851117]), 
    #     orientation=np.array([3.69097892e-04, 1.07786188e-01, 9.94168199e-01, 3.40492107e-03]),                                        
    # )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    packing_hiking_equipment_into_car(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
