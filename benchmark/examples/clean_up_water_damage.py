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

    
def clean_up_water_damage(scene_model, headless=False):

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
    cabinet = og.sim.scene.object_registry("name", "bottom_cabinet_no_top_qudfwe_0")
    dehumidifier = og.sim.scene.object_registry("name", "dehumidifier_189")
    disinfectant = og.sim.scene.system_registry("name", "disinfectant")
    disinfectant_bottle = og.sim.scene.object_registry("name", "disinfectant_bottle_188")
    floor = og.sim.scene.object_registry("name", "floors_sktjer_0")
    mold = og.sim.scene.system_registry("name", "mold")

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
    video_recorder.set(camera=og.sim.viewer_camera, robot = robot, save_path=f"{og.root_path}/../../images/clean_up_water_damage", name="clean_up_water_damage", trav_map_img=trav_map_img)

    position = dehumidifier.get_position() + np.array([-1.0, 0.0, 0.0])
    robot.set_position(position)
    # robot.set_position(np.array([-2.31474, 3.76517, robot.get_position()[2]]))
    # euler = T.quat2euler(carton.get_orientation())
    # quat = T.euler2quat(np.array([euler[0] + 90, euler[1], 0.0]))
    # # carton.get_orientation()
    # carton.set_position_orientation(carton.get_position() + np.array([0.0, 0.0, 0.5]), quat)

    # Expand the filed of view
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length = 4.0  


    for i in range(100):
        disinfectant_bottle.keep_still()
        video_recorder.get_video()
        og.sim.step()

    position = cabinet.get_position() + np.array([0.0, -1.0, 0.0])
    go_P(position, robot, trav_map, trav_map_size)

    cabinet.states[object_states.Open].set_value(True, True)
    for i in range(5):
        disinfectant_bottle.keep_still()
        video_recorder.get_video()
        og.sim.step()

    pick_A(disinfectant_bottle, robot)
    for i in range(5):
        disinfectant_bottle.set_position(robot.get_eef_position())
        video_recorder.get_video()
        og.sim.step()

    disinfectant.generate_particles_on_object(floor, instancer_idn=None, particle_group=0, sampling_distance=None, max_samples=800, min_samples_for_success=100, self_collision=True, prototype_indices=None)


    # carton_pos = carton.get_position()
    # place_A_at_P(bottle_of_liquid_soap, np.array([carton_pos[0], carton_pos[1], 0.5 * bottle_of_liquid_soap.native_bbox[2]]), robot)
    # # place_A_in_B(bottle_of_liquid_soap, carton, robot)

    # pick_A(bottle_of_medicine, robot)
    # place_A_at_P(bottle_of_medicine, np.array([carton_pos[0], carton_pos[1] + 0.1, 0.5 * bottle_of_medicine.native_bbox[2] + 0.05]), robot)
    # # place_A_in_B(bottle_of_medicine, carton, robot)

    # pick_A(bottle_of_vodka, robot)
    # place_A_at_P(bottle_of_vodka, np.array([carton_pos[0], carton_pos[1] + 0.15, 0.5 * bottle_of_vodka.native_bbox[2] + 0.05]), robot)
    # # place_A_in_B(bottle_of_vodka, carton, robot)

    # pick_A(box_of_chocolates, robot)
    # place_A_at_P(box_of_chocolates, np.array([carton_pos[0] - 0.15, carton_pos[1] , 0.5 * box_of_chocolates.native_bbox[2] + 0.05]), robot)
    # # place_A_in_B(box_of_chocolates, carton, robot)

    # # pick_A(cookie, robot)
    # # place_A_at_P(cookie, np.array([carton_pos[0] - 0.1, carton_pos[1], 0.5 * cookie.native_bbox[2]]), robot)
    # # # place_A_in_B(cookie, carton, robot)

    # pick_A(flashlight, robot)
    # place_A_at_P(flashlight, np.array([carton_pos[0] + 0.2, carton_pos[1] + 0.2, 0.5 * flashlight.native_bbox[2] + 0.05]), robot)
    # # place_A_in_B(flashlight, carton, robot)

    # pick_A(toothbrush, robot)
    # place_A_at_P(toothbrush, np.array([carton_pos[0] + 0.15, carton_pos[1] + 0.08, 0.5 * toothbrush.native_bbox[2] + 0.05]), robot)
    # # place_A_in_B(toothbrush, carton, robot)

    # # go_A(electric_refrigerator, robot, trav_map, trav_map_size)
    # # print("open electric_refrigerator")
    # # electric_refrigerator.states[object_states.Open].set_value(True,True)
    # # for i in range(5):
    # #     video_recorder.get_video()
    # #     og.sim.step()
    # # pick_A(brisket, robot)
    # # close_A_with_B(electric_refrigerator, brisket, robot)
        
    # # go_P_with_B(position, pie, robot, trav_map, trav_map_size)
    # # open_A_with_B(oven, pie, robot)
    # # place_A_in_B(pie, oven, robot)
    
    # # for i in range(20):
    while True:
        disinfectant_bottle.set_position(robot.get_eef_position())
        video_recorder.get_video()
        og.sim.step()

    # Release the video
    video_recorder.release()

def main(random_selection=False, headless=False, short_exec=False):
    
    config_filename = os.path.join(og.example_config_path, "fetch_behavior.yaml")
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    cfg["scene"]["scene_model"] = "Beechwood_0_int"
    cfg["task"]["activity_name"] = "clean_up_water_damage"

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Update the viewer camera's pose so that it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-4.88083544, -6.61283953,  2.39632796]), 
        orientation=np.array([4.14834077e-01, 4.96770317e-04, 1.08946220e-03, 9.09896288e-01]),
    )
   
    for obj in env.scene.systems:
        if any([name in obj.name for name in ["floor", "wall", "ceiling", "shelf", "picture"]]):
            continue
        print(obj.name,"   ", type(obj))

    clean_up_water_damage(cfg["scene"]["scene_model"])

    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    main()
