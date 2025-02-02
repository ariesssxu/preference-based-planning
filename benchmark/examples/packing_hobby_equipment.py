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
    
def packing_hobby_equipment(scene_model, headless=False):
    """
    (:goal 
        (and 
            (inside ?baseball.n.02_1 ?carton.n.02_1) 
            (inside ?soccer_ball.n.01_1 ?carton.n.02_1)  
            (forall 
                (?jigsaw_puzzle.n.01 - jigsaw_puzzle.n.01) 
                (inside ?jigsaw_puzzle.n.01 ?shelf.n.01_1)
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
    baseball = og.sim.scene.object_registry("name", "baseball_88")
    carton = og.sim.scene.object_registry("name", "carton_87")
    jigsaw_puzzle_1 = og.sim.scene.object_registry("name", "jigsaw_puzzle_85")
    jigsaw_puzzle_2 = og.sim.scene.object_registry("name", "jigsaw_puzzle_86")
    shelf = og.sim.scene.object_registry("name", "shelf_njwsoa_0")
    soccer_ball = og.sim.scene.object_registry("name", "soccer_ball_89")
    sofa = og.sim.scene.object_registry("name", "sofa_uixwiv_0")

    shelf_2 = og.sim.scene.object_registry("name", "shelf_owvfik_0")
    shelf_3 = og.sim.scene.object_registry("name", "shelf_owvfik_1")
    
    og.sim.remove_object(carton)
    # og.sim.remove_object(pad_3)

    og.sim.step()

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/packing_hobby_equipment", name="packing_hobby_equipment", trav_map_img=trav_map_img)

    # go and place baseball
    go_A(baseball, robot, trav_map, trav_map_size)
    pick_A(baseball, robot)
    go_A_with_B(shelf_2, baseball, robot, trav_map, trav_map_size)
    place_A_on_B(baseball, shelf_2, robot)
    print("inside baseball: ", shelf_2.states[object_states.Inside].get_value(baseball))

    # pick and place soccer ball
    pick_A(soccer_ball, robot)
    place_A_at_P(soccer_ball, shelf_2.get_position() + np.array([0.0, 0.0, soccer_ball.native_bbox[2] + soccer_ball.native_bbox[0]]), robot)
    
    # go to sofa and pick jigsaw_puzzle_1
    go_A(sofa, robot, trav_map, trav_map_size)
    pick_A(jigsaw_puzzle_1, robot)
    go_A_with_B(shelf, jigsaw_puzzle_1, robot, trav_map, trav_map_size)
    place_A_in_B(jigsaw_puzzle_1, shelf, robot)

    # go to sofa and pick jigsaw_puzzle_2
    go_A(sofa, robot, trav_map, trav_map_size)
    pick_A(jigsaw_puzzle_2, robot)
    go_A_with_B(shelf, jigsaw_puzzle_2, robot, trav_map, trav_map_size)
    place_A_at_P(jigsaw_puzzle_2, shelf.get_position() + np.array([0.0, 0.0, jigsaw_puzzle_2.native_bbox[2] * 0.5 + jigsaw_puzzle_2.native_bbox[0] * 0.5]), robot)

    for _ in range(20):
        video_recorder.get_video()
        og.sim.step()

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Benevolence_1_int"
    cfg["task"]["activity_name"] = "packing_hobby_equipment"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.06660735, -0.00247143,  2.2839962 ]), 
        orientation=np.array([0.15624997, 0.59385535, 0.76327615, 0.2008265]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    packing_hobby_equipment(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
