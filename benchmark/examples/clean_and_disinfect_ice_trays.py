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

    
def clean_and_disinfect_ice_trays(scene_model, headless=False):
    """
    (:goal 
        (and 
            (forall 
                (?tray.n.01 - tray.n.01)
                (and
                    (not 
                        (covered ?tray.n.01 ?adhesive_material.n.01_1)
                    )
                    (covered ?tray.n.01 ?disinfectant.n.01_1)
                )
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
    countertop = og.sim.scene.object_registry("name", "countertop_tpuwys_1")
    disinfectant_bottle = og.sim.scene.object_registry("name", "disinfectant_bottle_89")
    liquid_soap_bottle = og.sim.scene.object_registry("name", "liquid_soap_bottle_87")
    sponge = og.sim.scene.object_registry("name", "sponge_88")
    tray_1 = og.sim.scene.object_registry("name", "tray_85")
    tray_2 = og.sim.scene.object_registry("name", "tray_86")

    adhesive_material = og.sim.scene.system_registry("name", "adhesive_material")
    disinfectant = og.sim.scene.system_registry("name", "disinfectant")
    liquid_soap = og.sim.scene.system_registry("name", "liquid_soap")
    water = og.sim.scene.system_registry("name", "water")

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/clean_and_disinfect_ice_trays", name="clean_and_disinfect_ice_trays", trav_map_img=trav_map_img)

    liquid_soap.remove_all_particles()

    # 1. Take a few steps to let particles settle
    for _ in range(150):
        disinfectant_bottle.keep_still()
        liquid_soap_bottle.keep_still()
        video_recorder.get_video()
        og.sim.step()

    # 1.1 remove all particles
    # disinfectant.remove_all_particles()
    # og.sim.step()
    # input("Remove all disinfectant. Press [ENTER] to continue.")

    # 2. pour out disinfectant
    x_angle = (2 * np.pi * 1 / 12) 
    y_angle = (- 2 * np.pi * 1 / 12)   
    disinfectant_bottle_pos = disinfectant_bottle.get_position()
    quat = T.euler2quat(np.array([x_angle, y_angle, 0]))
    disinfectant_bottle.set_position_orientation(position=disinfectant_bottle_pos, orientation=quat)
    
    for i in range(25):
        og.sim.step()

    # input("Set the objects in place. Press [ENTER] to continue.")

    # 3. generate particles
    disinfectant.generate_particles_on_object(tray_2, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=100, min_samples_for_success=1, self_collision=True, prototype_indices=None)
    disinfectant.generate_particles_on_object(tray_1, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=50, min_samples_for_success=1, self_collision=True, prototype_indices=None)

    for i in range(15):
        og.sim.step()
    
    # 4. pick sponge and place it on tray_2
    go_A(countertop, robot, trav_map, trav_map_size)
    pick_A(sponge, robot)
    place_A_on_B(sponge, tray_2, robot)

    for i in range(15):
        og.sim.step()

    # 5. initial sponge position
    z_angle = (2 * np.pi * ( 55.674 - 44.638) / 90)   # tray_2 rotate [0.0, 0.0, -44.638]
    sponge_pos = sponge.get_position()
    quat = T.euler2quat(np.array([0, 0, z_angle]))
    sponge.set_position_orientation(position=sponge_pos, orientation=quat)
    
    for i in range(5):
        video_recorder.get_video()
        og.sim.step()

    sponge_pos = sponge.get_position()
    sponge.set_position(sponge_pos + np.array([-0.08, 0.27, 0.0]))
    for i in range(5):
        video_recorder.get_video()
        og.sim.step()

    # 6. Move sponge in square around tray_2
    deltas = [
        [35, np.array([0.01, -0.01, 0])],
        [10, np.array([-0.01, -0.01, 0])],
        [35, np.array([-0.01, 0.01, 0])],
        [10, np.array([-0.01, -0.01, 0])],
        [35, np.array([0.01, -0.01, 0])],
    ]

    for t, delta in deltas:
        for i in range(t):
            sponge.set_position(sponge.get_position() + delta)
            video_recorder.get_video()
            og.sim.step()

    while True:
        video_recorder.get_video()
        og.sim.step()

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Benevolence_1_int"
    cfg["task"]["activity_name"] = "clean_and_disinfect_ice_trays"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.73228131,  0.57160329,  1.8483926 ]), 
        orientation=np.array([0.37310115, -0.27903898, -0.52994404,  0.70858457]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    clean_and_disinfect_ice_trays(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
