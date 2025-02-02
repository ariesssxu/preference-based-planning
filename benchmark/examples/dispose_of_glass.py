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

    
def dispose_of_glass(scene_model, headless=False):
    

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
    floor = og.sim.scene.object_registry("name", "floors_shuuqu_0")
    recycling_bin = og.sim.scene.object_registry("name", "recycling_bin_594")
    shelf = og.sim.scene.object_registry("name", "wall_mounted_shelf_fjozhc_0")
    sink = og.sim.scene.object_registry("name", "sink_upwldu_0")
    water = og.sim.scene.system_registry("name", "water")
    water_glass_1 = og.sim.scene.object_registry("name", "water_glass_592")
    water_glass_2 = og.sim.scene.object_registry("name", "water_glass_593")

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

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    recycling_bin.set_position(recycling_bin.get_position() + np.array([-1.0, 0.0, 0.0])) 
    robot.set_position(np.array([23.9122, 20.7266, robot.get_position()[2]]))

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/dispose_of_glass", name="dispose_of_glass", trav_map_img=trav_map_img)

    # go_A(water_glass_1, robot, trav_map, trav_map_size)
    pick_A(water_glass_1, robot)
    position = robot.get_position() + np.array([-0.5, 0.0, 0.0])
    go_P_with_B(position, water_glass_1, robot, trav_map, trav_map_size)
    place_A_in_B(water_glass_1, recycling_bin, robot)

    for i in range(10):
        video_recorder.get_video()
        og.sim.step()

    # # Release the video
    video_recorder.release()


def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "house_single_floor"
    cfg["task"]["activity_name"] = "dispose_of_glass"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([23.75825614, 21.68932845,  3.24862202]),   
        orientation=np.array([0.00219156, 0.29298679, 0.95608723, 0.007151827]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    dispose_of_glass(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
