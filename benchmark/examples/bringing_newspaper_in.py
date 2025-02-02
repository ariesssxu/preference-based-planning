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

    
def bringing_newspaper_in(scene_model, headless=False):
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
    coffee_table = og.sim.scene.object_registry("name", "coffee_table_koagbh_0")
    drivery = og.sim.scene.object_registry("name", "driveway_qbxmgb_0")
    floor = og.sim.scene.object_registry("name", "floors_rhwsaa_0")
    newspaper = og.sim.scene.object_registry("name", "newspaper_276")
    door = og.sim.scene.object_registry("name", "door_vudhlc_2")

    ceiling_1 = og.sim.scene.object_registry("name", "ceilings_fzisrr_0")
    ceiling_2 = og.sim.scene.object_registry("name", "ceilings_pgwmer_0")
    wall_1 = og.sim.scene.object_registry("name", "walls_croxvf_0")
    wall_2 = og.sim.scene.object_registry("name", "walls_wnhkmr_0")
    wall_3 = og.sim.scene.object_registry("name", "walls_pcbfvo_0")
    wall_4 = og.sim.scene.object_registry("name", "walls_hnxiju_0")
    wall_5 = og.sim.scene.object_registry("name", "walls_sdxjmq_0")
    # door_1 = og.sim.scene.object_registry("name", "door_vudhlc_1")

    og.sim.remove_object(ceiling_1)
    og.sim.remove_object(ceiling_2)
    og.sim.remove_object(wall_1)
    og.sim.remove_object(wall_2)
    og.sim.remove_object(wall_3)
    og.sim.remove_object(wall_4)
    og.sim.remove_object(wall_5)
    # og.sim.remove_object(door_1)

    # robot.set_position_orientation([-13.3559, 4.91823, robot.get_position()[2]], robot.get_orientation())
    newspaper.set_position_orientation([9.75466, 5.31473, newspaper.get_position()[2]], newspaper.get_orientation())
    robot.set_position_orientation([9.41464, 6.89067, robot.get_position()[2]], robot.get_orientation())

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/bringing_newspaper_in", name="bringing_newspaper_in", trav_map_img=trav_map_img)
    
    # 1. pick newspaper
    for i in range(30):
        video_recorder.get_video()
        og.sim.step()

    pick_A(newspaper, robot)

    door.states[object_states.Open].set_value(True)
    for i in range(10):
        video_recorder.get_video()
        og.sim.step()
    og.sim.remove_object(door)

    # position = np.array([5.4017, 6.1504, 1.17623]) # inside door
    # go_P_with_B(position, newspaper, robot, trav_map, trav_map_size)
    # move_arm_to_target(get_arm_position(robot), robot)
    # og.sim.step()
    
    # near coffee table
    go_P_with_B(np.array([4.76824, 5.39209, robot.get_position()[2]]), newspaper, robot, trav_map, trav_map_size)
    move_arm_to_target(get_arm_position(robot), robot)
    og.sim.step()
    # input("Press [ENTER] to continue...")

    place_A_on_B(newspaper, coffee_table, robot)
    
    
    # while True:
    #     video_recorder.get_video()
    #     og.sim.step()

    for i in range(50):
        video_recorder.get_video()
        og.sim.step()

    # # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "house_double_floor_lower"
    cfg["task"]["activity_name"] = "bringing_newspaper_in"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        # position=np.array([-9.21329194, -2.35414824,  9.14161261]), 
        # orientation=np.array([3.40164722e-01, 6.34801118e-04, 1.75470275e-03, 9.40364015e-01]),
        # position=np.array([1.93751696, 7.52809171, 5.83693394]), 
        # orientation=np.array([-0.15616715,  0.32680658,  0.84101066, -0.40188351]),
        position=np.array([6.46417594, 0.77494804, 6.26015632]), 
        orientation=np.array([4.25575664e-01, -5.20186838e-04, -1.10624961e-03,  9.04922019e-01]),                                              
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    bringing_newspaper_in(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
