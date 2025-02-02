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

    
def cool_cakes(scene_model, headless=False):

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
    cookie_sheet = og.sim.scene.object_registry("name", "baking_sheet_189")
    countertop = og.sim.scene.object_registry("name", "countertop_tpuwys_2")
    floor = og.sim.scene.object_registry("name", "floors_sktjer_0")
    fruitcake = og.sim.scene.object_registry("name", "fruitcake_188")
    oven = og.sim.scene.object_registry("name", "oven_wuinhm_0")

    ceiling_1 = og.sim.scene.object_registry("name", "ceilings_ayfjzo_0")
    ceiling_2 = og.sim.scene.object_registry("name", "ceilings_gdodun_0")
    ceiling_3 = og.sim.scene.object_registry("name", "ceilings_gljdaz_0")
    ceiling_4 = og.sim.scene.object_registry("name", "ceilings_hzdavr_0")
    ceiling_5 = og.sim.scene.object_registry("name", "ceilings_kjpzku_0")
    ceiling_6 = og.sim.scene.object_registry("name", "ceilings_nkohbh_0")
    ceiling_7 = og.sim.scene.object_registry("name", "ceilings_rlsnwq_0")
    ceiling_8 = og.sim.scene.object_registry("name", "ceilings_uahaxk_0")
    ceiling_9 = og.sim.scene.object_registry("name", "ceilings_trhqld_0")
    ceiling_10 = og.sim.scene.object_registry("name", "ceilings_vjiuwu_0")
    ceiling_11 = og.sim.scene.object_registry("name", "ceilings_xqauyk_0")

    og.sim.remove_object(ceiling_1)
    og.sim.remove_object(ceiling_2)
    og.sim.remove_object(ceiling_3)
    og.sim.remove_object(ceiling_4)
    og.sim.remove_object(ceiling_5)
    og.sim.remove_object(ceiling_6)
    og.sim.remove_object(ceiling_7)
    og.sim.remove_object(ceiling_8)
    og.sim.remove_object(ceiling_9)
    og.sim.remove_object(ceiling_10)
    og.sim.remove_object(ceiling_11)
    # # water.remove_all_particles()

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/cool_cakes", name="cool_cakes", trav_map_img=trav_map_img)

    # robot.set_position(np.array([-2.31474, 3.76517, robot.get_position()[2]]))

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    go_A(oven, robot, trav_map, trav_map_size)
    print("open oven")
    oven.states[object_states.Open].set_value(True,True)
    for i in range(5):
        video_recorder.get_video()
        og.sim.step()

    pick_A(fruitcake, robot)
    # go_A_with_B(countertop, fruitcake, robot, trav_map, trav_map_size)
    position = np.array([-6.96472, -3.81266, 0.70252])
    go_P_with_B(position, fruitcake, robot, trav_map, trav_map_size)
    place_A_on_B(fruitcake, cookie_sheet, robot)


    while True:
        video_recorder.get_video()
        og.sim.step()

    # # Release the video

    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_0_int"
    cfg["task"]["activity_name"] = "cool_cakes"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-5.76556142, -6.16324014,  4.80192632]), 
        orientation=np.array([3.27193162e-01, -7.47716081e-04, -2.15966285e-03,  9.44954714e-01]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    cool_cakes(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
