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
from action import go_A, go_A_with_B, open_A, close_A, pick_A, place_A_on_B, place_A_in_B, toggle_on_A, toggle_off_A, wait_N, cook_A, cut_A_with_B, add_to_A_with_B, heat_A, close_A_with_B, open_A_with_B, place_A_at_P, uncover_A_with_B, clean_A_with_B, go_P_with_B
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

    
def spraying_for_bugs(scene_model, headless=False):
    """
        (and 
            (covered ?lawn.n.01_1 ?insectifuge.n.01_1) 
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
    floor = og.sim.scene.object_registry("name", "floor_uddigl_0")
    # insectifuge = og.sim.scene.system_registry("name", "insectifuge")
    insectifuge_atomizer = og.sim.scene.object_registry("name", "insectifuge_atomizer_276")
    lawn = og.sim.scene.object_registry("name", "lawn_ogzsvf_0")

    ceiling_1 = og.sim.scene.object_registry("name", "ceilings_fzisrr_0")
    ceiling_2 = og.sim.scene.object_registry("name", "ceilings_pgwmer_0")

    og.sim.remove_object(ceiling_1)
    og.sim.remove_object(ceiling_2)

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0

    # Create video generator
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/spraying_for_bugs", name="spraying_for_bugs", trav_map_img=trav_map_img)
    
    
    while True:
        video_recorder.get_video()
        insectifuge_atomizer.keep_still()
        og.sim.step()

    # 1. take away the bag of rubbish 1
    go_A(insectifuge_atomizer, robot, trav_map, trav_map_size)
    pick_A(insectifuge_atomizer, robot)
    go_A_with_B(lawn, insectifuge_atomizer, robot, trav_map, trav_map_size)

    # insectifuge.generate_particles_on_object(lawn, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=300, min_samples_for_success=1, self_collision=True, prototype_indices=None)
    


    for i in range(5):
        video_recorder.get_video()
        og.sim.step()

    # # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "house_double_floor_lower"
    cfg["task"]["activity_name"] = "spraying_for_bugs"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.93751696, 7.52809171, 5.83693394]), 
        orientation=np.array([-0.15616715,  0.32680658,  0.84101066, -0.40188351]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    spraying_for_bugs(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
