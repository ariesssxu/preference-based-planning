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

    
def changing_dogs_water(scene_model, headless=False):

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
    bowl = og.sim.scene.object_registry("name", "bowl_188")
    floors = og.sim.scene.object_registry("name", "floors_gjpvfd_0")
    sink = og.sim.scene.object_registry("name", "sink_czyfhq_0")
    water = og.sim.scene.system_registry("name", "water")
    straight_chair = og.sim.scene.object_registry("name", "straight_chair_vkgbbl_1")
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet_no_top_qohxjq_1")

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
    wall_1 = og.sim.scene.object_registry("name", "walls_tjfjwe_0")

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
    og.sim.remove_object(wall_1)
    # # water.remove_all_particles()

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/changing_dogs_water", name="changing_dogs_water", trav_map_img=trav_map_img)

    position = bottom_cabinet.get_position() + np.array([1.0, 0.0, 0.0])
    robot.set_position(position)
    # robot.set_position(np.array([-2.31474, 3.76517, robot.get_position()[2]]))
    # euler = T.quat2euler(carton.get_orientation())
    # quat = T.euler2quat(np.array([euler[0] + 90, euler[1], 0.0]))
    # # carton.get_orientation()
    # carton.set_position_orientation(carton.get_position() + np.array([0.0, 0.0, 0.5]), quat)
    bowl.set_position(np.array([straight_chair.get_position()[0] - 0.2, straight_chair.get_position()[1] + 0.8, bowl.get_position()[2]]))
    origin_bowl_pos = bowl.get_position()
    origin_robot_pos = robot.get_position()

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0  

    pick_A(bowl, robot)
    go_A_with_B(sink, bowl, robot, trav_map, trav_map_size)
    print("put bowl on sink: ")

    position = sink.get_position() + np.array([0.15, -0.02, 0.35])
    place_A_at_P(bowl, position, robot)
    for i in range(20):
        video_recorder.get_video()
        og.sim.step()

    # sink.states[object_states.ToggledOn].set_value(True)
    water.generate_particles_on_object(bowl, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=180, min_samples_for_success=10, self_collision=True, prototype_indices=None)
    for i in range(100):
        video_recorder.get_video()
        og.sim.step()
    # sink.states[object_states.ToggledOn].set_value(False)
    
    water.remove_all_particles()
    pick_A(bowl, robot)
    go_P_with_B(origin_robot_pos, bowl, robot, trav_map, trav_map_size)
    place_A_at_P(bowl, origin_bowl_pos, robot)
    water.generate_particles_on_object(bowl, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=180, min_samples_for_success=10, self_collision=True, prototype_indices=None)
    
    for i in range(50):
    # while True:
        video_recorder.get_video()
        og.sim.step()

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_0_int"
    cfg["task"]["activity_name"] = "changing_dogs_water"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-5.42231396, -2.70150896,  4.09385132]), 
        orientation=np.array([0.00098377, 0.27699775, 0.960864  , 0.00341278]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    changing_dogs_water(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
