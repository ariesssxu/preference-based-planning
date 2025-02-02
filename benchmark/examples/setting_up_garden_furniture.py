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

    
def setting_up_garden_furniture(scene_model, headless=False):

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
    coffee_table = og.sim.scene.object_registry("name", "coffee_table_wgtaip_0")
    driveway = og.sim.scene.object_registry("name", "driveway_hmbdky_0")
    floor = og.sim.scene.object_registry("name", "floors_khjmrr_0")
    lawn = og.sim.scene.object_registry("name", "lawn_wwoqjw_0")
    lawn_chair_1 = og.sim.scene.object_registry("name", "garden_chair_593")
    lawn_chair_2 = og.sim.scene.object_registry("name", "garden_chair_594")
    lawn_chair_3 = og.sim.scene.object_registry("name", "garden_chair_595")
    pitcher = og.sim.scene.object_registry("name", "pitcher_592")
    teacup_1 = og.sim.scene.object_registry("name", "teacup_596")
    teacup_2 = og.sim.scene.object_registry("name", "teacup_597")
    tree = og.sim.scene.object_registry("name", "tree_flkzbo_4")

    ceiling_1 = og.sim.scene.object_registry("name", "ceilings_apkslx_0")
    ceiling_2 = og.sim.scene.object_registry("name", "ceilings_awbqsa_0")
    ceiling_3 = og.sim.scene.object_registry("name", "ceilings_frqekm_0")
    ceiling_4 = og.sim.scene.object_registry("name", "ceilings_gasyop_0")
    ceiling_5 = og.sim.scene.object_registry("name", "ceilings_ibxzsn_0")
    ceiling_6 = og.sim.scene.object_registry("name", "ceilings_iskjpi_0")
    ceiling_7 = og.sim.scene.object_registry("name", "ceilings_jjyrdh_0")
    ceiling_8 = og.sim.scene.object_registry("name", "ceilings_khkogf_0")
    ceiling_9 = og.sim.scene.object_registry("name", "ceilings_kxjvpr_0")
    ceiling_10 = og.sim.scene.object_registry("name", "ceilings_nsysju_0")
    ceiling_11 = og.sim.scene.object_registry("name", "ceilings_qdlnli_0")
    ceiling_12 = og.sim.scene.object_registry("name", "ceilings_qenicw_0")
    ceiling_13 = og.sim.scene.object_registry("name", "ceilings_shvqqe_0")
    ceiling_14 = og.sim.scene.object_registry("name", "ceilings_tbhzzb_0")
    ceiling_15 = og.sim.scene.object_registry("name", "ceilings_tcwyvb_0")
    ceiling_15 = og.sim.scene.object_registry("name", "ceilings_tjwkje_0")
    ceiling_16 = og.sim.scene.object_registry("name", "ceilings_wbbkuv_0")
    roof_1 = og.sim.scene.object_registry("name", "roof_amxayl_0")
    roof_2 = og.sim.scene.object_registry("name", "roof_jqqtaq_0")

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
    og.sim.remove_object(roof_1)
    og.sim.remove_object(roof_2)
    
    robot.set_position(lawn_chair_2.get_position() + np.array([-1.0, 1.0, 0.0]))
    og.sim.step()

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/setting_up_garden_furniture", name="setting_up_garden_furniture", trav_map_img=trav_map_img)

    pick_A(lawn_chair_2, robot)
    position = lawn_chair_3.get_position() + np.array([-0.5, -2.5, 0.0])
    go_P_with_B(position, lawn_chair_2, robot, trav_map, trav_map_size)
    position = lawn_chair_3.get_position() + np.array([-2.0, -1.5, 0.0])
    place_A_at_P(lawn_chair_2, position, robot)

    position = lawn_chair_1.get_position() + np.array([-1.0, -1.0, 0.0])
    go_P(position, robot, trav_map, trav_map_size)
    pick_A(lawn_chair_1, robot)
    position = lawn_chair_3.get_position() + np.array([2.5, -1.0, 0.0])
    go_P_with_B(position, lawn_chair_1, robot, trav_map, trav_map_size)
    position = lawn_chair_3.get_position() + np.array([1.5, -1.0, 0.0])
    place_A_at_P(lawn_chair_1, position, robot)

    z_angle = (2 * np.pi * (-90) / 90)   
    lawn_chair_pos = lawn_chair_1.get_position()
    quat = T.euler2quat(np.array([0, 0, z_angle]))
    lawn_chair_1.set_position_orientation(position=lawn_chair_pos, orientation=quat)


    while True:
        video_recorder.get_video()
        og.sim.step()

    # 1. pick envelope_1
    # position = np.array([2.33961, 3.81885, 1.17623]) # inside door
    # go_P(position, robot, trav_map, trav_map_size)
    # pick_A(envelope_3, robot)

    # door_1_pos = door_1.get_position()
    # go_P_with_B(door_1_pos + np.array([0.0, 2.0, 0.0]), envelope_3, robot, trav_map, trav_map_size)
    # door_1.states[object_states.Open].set_value(True)
    
    # for i in range(10):
    #     video_recorder.get_video()
    #     og.sim.step()
    # # og.sim.remove_object(door_1)
    
    # go_P_with_B(door_1_pos + np.array([0.0, -1.0, 0.0]), envelope_3, robot, trav_map, trav_map_size)
    # go_P_with_B(np.array([-4.91862, -2.52522, 0.28975]), envelope_3, robot, trav_map, trav_map_size)
    # go_P_with_B(np.array([-6.5965, -2.71411, 0.28975]), envelope_3, robot, trav_map, trav_map_size)
    # place_A_on_B(envelope_3, mailbox, robot)

    # for i in range(50):
    #     video_recorder.get_video()
    #     og.sim.step()

    # # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["task"]["activity_name"] = "setting_up_garden_furniture"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-22.69222722,  -1.02314673,   5.13288252]),  
        orientation=np.array([0.47346337, -0.22515921, -0.365713  ,  0.76901871]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    setting_up_garden_furniture(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
