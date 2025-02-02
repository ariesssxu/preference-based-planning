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
    
def set_up_a_bird_cage(scene_model, headless=False):

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
    water = og.sim.scene.system_registry("name", "water")
    apple = og.sim.scene.object_registry("name", "apple_193")
    birdcage = og.sim.scene.object_registry("name", "birdcage_188")
    bowl_1 = og.sim.scene.object_registry("name", "bowl_189")
    bowl_2 = og.sim.scene.object_registry("name", "bowl_190")
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table_dnsjnv_0")
    floors = og.sim.scene.object_registry("name", "floors_qtdpcm_0")
    sink = og.sim.scene.object_registry("name", "sink_czyfhq_0")
    toy_figure_1 = og.sim.scene.object_registry("name", "toy_figure_191")
    toy_figure_2 = og.sim.scene.object_registry("name", "toy_figure_192")
    straight_chair_1 = og.sim.scene.object_registry("name", "straight_chair_dmcixv_0")

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
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/set_up_a_bird_cage", name="set_up_a_bird_cage", trav_map_img=trav_map_img)

    bowl_1.set_position(np.array([bowl_1.get_position()[0], bowl_1.get_position()[1], bowl_1.get_position()[2] + 0.1]))
    bowl_2.set_position(np.array([bowl_2.get_position()[0], bowl_2.get_position()[1], bowl_2.get_position()[2] + 0.1]))
    robot.set_position(straight_chair_1.get_position() + np.array([0.0, -0.6, 0.0]))

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0  

    # pick_A(bowl_1, robot)
    # position = birdcage.get_position() + np.array([- 0.07, 0.0, 0.0])
    # # place_A_at_P(bowl_1, position, robot)

    # robot_rotate_with_A(robot, position, bowl_1)
    # for _ in range(10):
    #     bowl_1.set_position(robot.get_eef_position())
    #     video_recorder.get_video()
    #     og.sim.step()
    # arm_pos = position
    # for _ in range(10):
    #     bowl_1.set_position_orientation(
    #         position=arm_pos,
    #         orientation=[0, 0, 0, 1],
    #     )
    #     bowl_1.keep_still()
    #     birdcage.keep_still()
    #     video_recorder.get_video()
    #     og.sim.step()

    # for i in range(50):
    #     birdcage.keep_still()
    #     bowl_1.keep_still()
    #     video_recorder.get_video()
    #     og.sim.step()

    # water.generate_particles_on_object(bowl_1, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=50, min_samples_for_success=10, self_collision=True, prototype_indices=None)
    # for i in range(50):
    #     birdcage.keep_still()
    #     bowl_1.keep_still()
    #     video_recorder.get_video()
    #     og.sim.step()

    pick_A(apple, robot)
    position = birdcage.get_position() + np.array([- 0.02, 0.02, 0.02])
    place_A_at_P(apple, position, robot)
    for i in range(50):
        birdcage.keep_still()
        apple.keep_still()
        video_recorder.get_video()
        og.sim.step()

    # pick_A(toy_figure_2, robot)
    # robot_rotate(robot, toy_figure_2)
    end = get_robot_orientation(toy_figure_2.get_position(), robot.get_position())
    start = robot.get_orientation()
    position = robot.get_position()
    rotate_list = rotate_interpolation(start, end, 10)
    for ori in rotate_list:
        robot.keep_still()
        robot.set_orientation(ori)
        birdcage.keep_still()
        apple.keep_still()
        for _ in range(10):
            robot.set_position(position)
            video_recorder.get_video(text=f"rotate")
            robot.keep_still()
            birdcage.keep_still()
            apple.keep_still()
            og.sim.step()
    for _ in range(10):
        toy_figure_2.set_position_orientation(
            position=robot.get_eef_position(),
            orientation=[0, 0, 0, 1],
        )
        toy_figure_2.keep_still()
        birdcage.keep_still()
        apple.keep_still()
        video_recorder.get_video(text=f"pick {toy_figure_2.name}")
        og.sim.step()


    position = birdcage.get_position() + np.array([ - 0.035, 0.01, 0.02])
    # place_A_at_P(toy_figure_2, position, robot)
    # robot_rotate_with_A(robot, position, toy_figure_2)
    end = get_robot_orientation(position, robot.get_position())
    start = robot.get_orientation()
    position = robot.get_position()
    rotate_list = rotate_interpolation(start, end, 10)
    for ori in rotate_list:
        robot.keep_still()
        robot.set_orientation(ori)
        robot.release_grasp_immediately()
        birdcage.keep_still()
        apple.keep_still()
        for _ in range(10):
            robot.set_position(position)
            toy_figure_2.set_position_orientation(
                position=robot.get_eef_position(),
                orientation=[0, 0, 0, 1],
            )
            # Keep things grasped not dropping
            toy_figure_2.keep_still()
            birdcage.keep_still()
            apple.keep_still()
            robot.release_grasp_immediately()
            video_recorder.get_video(text=f"rotate with {toy_figure_2.name}")
            robot.keep_still()
            og.sim.step()

    for _ in range(10):
        toy_figure_2.set_position(robot.get_eef_position())
        birdcage.keep_still()
        apple.keep_still()
        video_recorder.get_video()
        og.sim.step()
    position = birdcage.get_position() + np.array([ - 0.03, 0.01, 0.015])
    for _ in range(10):
        toy_figure_2.set_position_orientation(
            position=position,
            orientation=[0, 0, 0, 1],
        )
        toy_figure_2.keep_still()
        birdcage.keep_still()
        apple.keep_still()
        video_recorder.get_video()
        og.sim.step()

    for i in range(50):
        birdcage.keep_still()
        apple.keep_still()
        toy_figure_2.keep_still()
        video_recorder.get_video()
        og.sim.step()

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_0_int"
    cfg["task"]["activity_name"] = "set_up_a_bird_cage"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-7.04432074, -1.03432547,  3.62315318]),  
        orientation=np.array([0.24454049, 0.23899809, 0.65682499, 0.67205713]),
        # position=np.array([-9.09306832, -1.2597092 ,  1.24127606]),   
        # orientation=np.array([0.00736107, 0.43726661, 0.8991744 , 0.01513711]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    set_up_a_bird_cage(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
