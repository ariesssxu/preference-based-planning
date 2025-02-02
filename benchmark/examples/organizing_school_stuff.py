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
from action import go_A, go_A_with_B, open_A, close_A, pick_A, place_A_on_B, place_A_in_B, toggle_on_A, toggle_off_A, wait_N, cook_A, cut_A_with_B, add_to_A_with_B, heat_A, close_A_with_B, open_A_with_B, place_A_at_P, uncover_A_with_B, clean_A_with_B
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

    
def organizing_school_stuff(scene_model, headless=False):
    """
    (:goal 
        (and 
            (and 
                (nextto ?folder.n.02_1 ?book.n.02_1) 
                (nextto ?folder.n.02_1 ?backpack.n.01_1) 
                (nextto ?book.n.02_1 ?backpack.n.01_1)
            ) 
            (inside ?marker.n.03_1 ?backpack.n.01_1)
            (inside ?pencil.n.01_1 ?backpack.n.01_1) 
            (inside ?pen.n.01_1 ?backpack.n.01_1) 
            (inside ?calculator.n.02_1 ?backpack.n.01_1) 
            (ontop ?backpack.n.01_1 ?bed.n.01_1)
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
    backpack = og.sim.scene.object_registry("name", "backpack_88")
    bed = og.sim.scene.object_registry("name", "bed_fnzxxr_0")
    book = og.sim.scene.object_registry("name", "paperback_book_86")
    calculator = og.sim.scene.object_registry("name", "calculator_85")
    folder = og.sim.scene.object_registry("name", "folder_87")
    marker = og.sim.scene.object_registry("name", "marker_82")
    pen = og.sim.scene.object_registry("name", "pen_84")
    pencil = og.sim.scene.object_registry("name", "colored_pencil_83")
    table = og.sim.scene.object_registry("name", "breakfast_table_bmnubh_0")
    
    bottom_cabinet = og.sim.scene.object_registry("name", "bottom_cabinet_bamfsz_0")

    ceiling_1 = og.sim.scene.object_registry("name", "ceilings_adtedr_0")
    ceiling_2 = og.sim.scene.object_registry("name", "ceilings_gpivsz_0")
    ceiling_3 = og.sim.scene.object_registry("name", "ceilings_ouguhp_0")
    ceiling_4 = og.sim.scene.object_registry("name", "ceilings_plnpyp_0")
    ceiling_5 = og.sim.scene.object_registry("name", "ceilings_xrasdh_0")
    ceiling_6 = og.sim.scene.object_registry("name", "ceilings_zbmcak_0")
    ceiling_7 = og.sim.scene.object_registry("name", "ceilings_zplkyb_0")

    og.sim.remove_object(ceiling_1)
    og.sim.remove_object(ceiling_2)
    og.sim.remove_object(ceiling_3)
    og.sim.remove_object(ceiling_4)
    og.sim.remove_object(ceiling_5)
    og.sim.remove_object(ceiling_6)
    og.sim.remove_object(ceiling_7)
    # og.sim.remove_object(bottom_cabinet)

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/organizing_school_stuff", name="organizing_school_stuff", trav_map_img=trav_map_img)

    # 0. wait for everything to settle
    for _ in range(50):
        bottom_cabinet.keep_still()  
        video_recorder.get_video()
        og.sim.step()



    # 1. put backpack next to folder and book
    go_A(bed, robot, trav_map, trav_map_size)
    pick_A(backpack, robot)
    # place_A_at_P(backpack, 0.5 * (book.get_position() + folder.get_position()), robot)
    place_A_on_B(backpack, bed, robot)
    
    for i in range(25):
        video_recorder.get_video()
        og.sim.step()

    input("Set the objects in place. Press [ENTER] to continue.")
    
    # while True:
    #     video_recorder.get_video()
    #     og.sim.step()

    # 3.place marker in backpack
    go_A(marker, robot, trav_map, trav_map_size)
    pick_A(marker, robot)
    place_A_in_B(marker, bed, robot)

    for i in range(150):
        video_recorder.get_video()
        og.sim.step()
    input("Set the objects in place. Press [ENTER] to continue.")

    # 3.place pen in backpack
    go_A(pen, robot, trav_map, trav_map_size)
    pick_A(pen, robot)
    place_A_in_B(pen, bed, robot)

    for i in range(150):
        video_recorder.get_video()
        og.sim.step()
    input("Set the objects in place. Press [ENTER] to continue.")

    # 3.place calculator in backpack
    go_A(calculator, robot, trav_map, trav_map_size)
    pick_A(calculator, robot)
    place_A_in_B(calculator, bed, robot)

    for i in range(150):
        video_recorder.get_video()
        og.sim.step()
    input("Set the objects in place. Press [ENTER] to continue.")

    while True:
        video_recorder.get_video()
        og.sim.step()

    # # 5. initial sponge position
    # z_angle = (2 * np.pi * ( 55.674 - 44.638) / 90)   # tray_2 rotate [0.0, 0.0, -44.638]
    # sponge_pos = sponge.get_position()
    # quat = T.euler2quat(np.array([0, 0, z_angle]))
    # sponge.set_position_orientation(position=sponge_pos, orientation=quat)
    
    # for i in range(5):
    #     video_recorder.get_video()
    #     og.sim.step()

    # sponge_pos = sponge.get_position()
    # sponge.set_position(sponge_pos + np.array([-0.08, 0.27, 0.0]))
    # for i in range(5):
    #     video_recorder.get_video()
    #     og.sim.step()

    # # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Benevolence_2_int"
    cfg["task"]["activity_name"] = "organizing_school_stuff"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.05822169, -0.87566147,  3.5714498 ]), 
        orientation=np.array([1.87315239e-01, -7.37959251e-04, -3.86997235e-03,  9.82291952e-01]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    organizing_school_stuff(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
