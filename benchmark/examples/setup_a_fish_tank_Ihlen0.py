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

    
def setup_a_fish_tank(scene_model, headless=False):

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
    bucket = og.sim.scene.object_registry("name", "bucket_89")
    floors = og.sim.scene.object_registry("name", "floors_lwmene_0")
    pebble = og.sim.scene.object_registry("name", "pebble_91")
    coffee_table = og.sim.scene.object_registry("name", "coffee_table_fqluyq_0")
    tank = og.sim.scene.object_registry("name", "tank_88")
    water = og.sim.scene.system_registry("name", "water")
    water_filter = og.sim.scene.object_registry("name", "water_filter_90")

    armchair = og.sim.scene.object_registry("name", "armchair_tcxiue_2")


    ceiling_1 = og.sim.scene.object_registry("name", "ceilings_dashlx_0")
    ceiling_2 = og.sim.scene.object_registry("name", "ceilings_dslvvl_0")
    ceiling_3 = og.sim.scene.object_registry("name", "ceilings_eanlkl_0")
    ceiling_4 = og.sim.scene.object_registry("name", "ceilings_gjwnre_0")
    ceiling_5 = og.sim.scene.object_registry("name", "ceilings_ndlpeo_0")
    ceiling_6 = og.sim.scene.object_registry("name", "ceilings_rnaypg_0")
    ceiling_7 = og.sim.scene.object_registry("name", "ceilings_vubcbl_0")

    og.sim.remove_object(ceiling_1)
    og.sim.remove_object(ceiling_2)
    og.sim.remove_object(ceiling_3)
    og.sim.remove_object(ceiling_4)
    og.sim.remove_object(ceiling_5)
    og.sim.remove_object(ceiling_6)
    og.sim.remove_object(ceiling_7)
    water.remove_all_particles()


    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/setup_a_fish_tank_Ihlen0", name="setup_a_fish_tank_Ihlen0", trav_map_img=trav_map_img)

    robot.set_position(np.array([-1.34008, 3.47453, robot.get_position()[2]]))
    # [armchair.get_position()[0] - 1.5, armchair.get_position()[1] - 1.0, robot.get_position()[2]])

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # wait for system to settle
    for i in range(10):
        video_recorder.get_video()
        bucket.keep_still()
        og.sim.step()

    # position = np.array([-1.04008, 3.47453, robot.get_position()[2]])
    # go_P(position, robot, trav_map, trav_map_img)

    print("open lid")
    tank.states[object_states.Open].set_value(True, True)
    for i in range(10):
        video_recorder.get_video()
        bucket.keep_still()
        og.sim.step()
    print("finish open lid")

    pick_A(bucket, robot)
    for i in range(20):
        video_recorder.get_video()
        bucket.keep_still()
        robot.keep_still()
        og.sim.step()

    water.generate_particles_on_object(tank, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=800, min_samples_for_success=10, self_collision=True, prototype_indices=None)

    for i in range(50):
        video_recorder.get_video()
        tank.keep_still()
        bucket.keep_still()
        robot.keep_still()
        og.sim.step()

    # # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Ihlen_0_int"
    cfg["task"]["activity_name"] = "setup_a_fish_tank"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.42199794, 2.50082071, 4.71524219]), 
        orientation=np.array([0.14207184, 0.14093674, 0.69001866, 0.69557651]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    setup_a_fish_tank(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
