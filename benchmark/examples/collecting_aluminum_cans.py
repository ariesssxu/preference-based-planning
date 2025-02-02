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
    
def collecting_aluminum_cans(scene_model, headless=False):

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
    bed = og.sim.scene.object_registry("name", "bed_zrumze_0")
    bucket = og.sim.scene.object_registry("name", "ice_bucket_189")
    can_of_soda_1 = og.sim.scene.object_registry("name", "can_of_soda_183")
    can_of_soda_2 = og.sim.scene.object_registry("name", "can_of_soda_184")
    can_of_soda_3 = og.sim.scene.object_registry("name", "can_of_soda_185")
    can_of_soda_4 = og.sim.scene.object_registry("name", "can_of_soda_186")
    can_of_soda_5 = og.sim.scene.object_registry("name", "can_of_soda_187")
    can_of_soda_6 = og.sim.scene.object_registry("name", "can_of_soda_188")
    floor = og.sim.scene.object_registry("name", "floors_nzidpe_0")
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet_nddvba_0")
    floor_lamp = og.sim.scene.object_registry("name", "floor_lamp_vdxlda_0")

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
    og.sim.remove_object(can_of_soda_2)
    og.sim.remove_object(can_of_soda_3)
    og.sim.remove_object(floor_lamp)
    # # water.remove_all_particles()

    euler = T.quat2euler(robot.get_orientation())
    quat = T.euler2quat(np.array([euler[0], euler[1], euler[2] - 90]))
    position = bottom_cabinet.get_position() + np.array([1.1, 0.1, 0.0])
    robot.set_position_orientation(position, quat)

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/collecting_aluminum_cans", name="collecting_aluminum_cans", trav_map_img=trav_map_img)

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0  

    pick_A(can_of_soda_5, robot)
    place_A_in_B(can_of_soda_5, bucket, robot)

    position = bottom_cabinet.get_position() + np.array([1.2, 0.7, 0.0])
    go_P(position, robot, trav_map, trav_map_img)
    pick_A(can_of_soda_6, robot)
    position = bottom_cabinet.get_position() + np.array([1.2, 0.1, 0.0])
    go_P_with_B(position, can_of_soda_6, robot, trav_map, trav_map_img)
    place_A_in_B(can_of_soda_6, bucket, robot)

    position = can_of_soda_4.get_position() + np.array([- 0.9, 0.9, 0.0])
    go_P(position, robot, trav_map, trav_map_img)
    pick_A(can_of_soda_4, robot)
    place_A_in_B(can_of_soda_4, bucket, robot)
    pick_A(can_of_soda_1, robot)
    place_A_in_B(can_of_soda_1, bucket, robot)

    for i in range(50):
        video_recorder.get_video()
        og.sim.step()

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_1_int"
    cfg["task"]["activity_name"] = "collecting_aluminum_cans"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-7.17221132, -3.47501177,  5.35141454]),  
        orientation=np.array([0.10203284, -0.10164924, -0.6984157 ,  0.70105081]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    collecting_aluminum_cans(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
