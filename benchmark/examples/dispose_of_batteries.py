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
    
def dispose_of_batteries(scene_model, headless=False):

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
    ashcan = og.sim.scene.object_registry("name", "trash_can_185")
    battery_1 = og.sim.scene.object_registry("name", "battery_183")
    battery_2 = og.sim.scene.object_registry("name", "battery_184")
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet_jhymlr_0")
    floor = og.sim.scene.object_registry("name", "floors_nzidpe_0")
    hand_blower = og.sim.scene.object_registry("name", "hair_dryer_187")
    razor = og.sim.scene.object_registry("name", "razor_186")

    ceiling_1 = og.sim.scene.object_registry("name", "ceilings_bpheuv_0")
    ceiling_2 = og.sim.scene.object_registry("name", "ceilings_bwcjal_0")
    ceiling_3 = og.sim.scene.object_registry("name", "ceilings_cmtgio_0")
    ceiling_4 = og.sim.scene.object_registry("name", "ceilings_hkpbtx_0")
    ceiling_5 = og.sim.scene.object_registry("name", "ceilings_holqmb_0")
    ceiling_6 = og.sim.scene.object_registry("name", "ceilings_jkgiww_0")
    ceiling_7 = og.sim.scene.object_registry("name", "ceilings_msnxsu_0")
    ceiling_8 = og.sim.scene.object_registry("name", "ceilings_odcayc_0")
    ceiling_9 = og.sim.scene.object_registry("name", "ceilings_pueokn_0")
    ceiling_10 = og.sim.scene.object_registry("name", "ceilings_qsqnsq_0")
    ceiling_11 = og.sim.scene.object_registry("name", "ceilings_sxfdbk_0")
    ceiling_12 = og.sim.scene.object_registry("name", "ceilings_unxwzc_0")
    ceiling_13 = og.sim.scene.object_registry("name", "ceilings_xcvxqs_0")
    ceiling_14 = og.sim.scene.object_registry("name", "ceilings_xqhgay_0")
    ceiling_15 = og.sim.scene.object_registry("name", "ceilings_yeqcqn_0")
    ceiling_16 = og.sim.scene.object_registry("name", "ceilings_ytbprd_0")
    bottom_cabinet_2 = og.sim.scene.object_registry("name", "bottom_cabinet_nddvba_0")
    # wall_1 = og.sim.scene.object_registry("name", "walls_fuuaks_0")
    # wall_2 = og.sim.scene.object_registry("name", "walls_dmusnk_0")
    # wall_3 = og.sim.scene.object_registry("name", "walls_qfsjoy_0")
    # wall_4 = og.sim.scene.object_registry("name", "walls_uwbxiv_0")
    # mirror = og.sim.scene.object_registry("name", "mirror_pevdqu_3")

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
    og.sim.remove_object(ceiling_12)
    og.sim.remove_object(ceiling_13)
    og.sim.remove_object(ceiling_14)
    og.sim.remove_object(ceiling_15)
    og.sim.remove_object(ceiling_16)
    # og.sim.remove_object(hand_blower)
    # og.sim.remove_object(razor)

    # euler = T.quat2euler(robot.get_orientation())
    # quat = T.euler2quat(np.array([euler[0], euler[1], euler[2] - 90]))
    # position = bottom_cabinet.get_position() + np.array([1.0, 0.1, 0.0])
    # position = mousetrap_1.get_position() + np.array([1.0, 0.0, 0.0])
    # origin_pos = position
    cabinet_pos = bottom_cabinet.get_position()
    position = cabinet_pos + np.array([-1.0, 0.0, 0.0])
    robot.set_position(position)

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/dispose_of_batteries", name="dispose_of_batteries", trav_map_img=trav_map_img)

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0  

    for i in range(50):
        bottom_cabinet.set_position(cabinet_pos)
        bottom_cabinet.keep_still()
        battery_2.keep_still()
        video_recorder.get_video()
        og.sim.step()

    pick_A(battery_2, robot)
    ashcan_pos = ashcan.get_position() + np.array([1.0, 0.5, 0.0])
    go_P_with_B(ashcan_pos, battery_2, robot, trav_map, trav_map_img)
    place_A_in_B(battery_2, ashcan, robot)

    position = bottom_cabinet_2.get_position() + np.array([1.1, 0.1, 0.0])
    go_P(position, robot, trav_map, trav_map_img)
    pick_A(battery_1, robot)
    go_P_with_B(ashcan_pos, battery_1, robot, trav_map, trav_map_img)
    place_A_in_B(battery_1, ashcan, robot)

    for i in range(50):
        video_recorder.get_video()
        og.sim.step()

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_1_int"
    cfg["task"]["activity_name"] = "dispose_of_batteries"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-7.88274633, -4.0272428 ,  5.20101094]),   
        orientation=np.array([-0.18446635,  0.18485714,  0.68329072, -0.68184587]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    dispose_of_batteries(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
