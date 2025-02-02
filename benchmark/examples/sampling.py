# open a scene and show all objects in it.
import time
import numpy as np
import os
import yaml
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_og_scenes, get_og_scene_path
from omnigibson.maps import SegmentationMap, TraversableMap
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

# Make sure object states, GPU dynamics, and transition rules are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = True


def convert(xy, resolution=0.1, size=200):
    return (xy / resolution + size / 2.0)

def load_task_config_yaml(config_path):
    with open(config_path, "r") as f:
        task_config = yaml.load(f, Loader=yaml.FullLoader)
    return task_config

def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates a BEHAVIOR Task environment in an online fashion.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 +
                "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Use a pre-sampled cached BEHAVIOR activity scene"
    # Choose the scene model to load
    scene_type = "InteractiveTraversableScene"
    scenes = get_available_og_scenes()
    # scene_model = choose_from_options(
    #     options=scenes, name="scene model", random_selection=random_selection)

    scene_model = "Beechwood_0_int"

    robot0_cfg = dict(
        type="Fetch",
        # we're just doing a grasping demo so we don't need all observation modalities
        obs_modalities=["rgb"],
        action_type="continuous",
        action_normalize=True,
        grasping_mode="physical",
    )

    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
        },
        "robots": [robot0_cfg],
    }

    # for quick load
    cfg["scene"]["load_object_categories"] = [
        "floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()

    og_dataset_path = gm.DATASET_PATH
    og_scenes_path = os.path.join(og_dataset_path, "scenes", scene_model)

    # test segmentation map
    map = SegmentationMap(scene_dir=og_scenes_path)
    print(map.get_room_type_by_point([0, 0]))

    

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()


# # # list all available objects
#     # Choose the scene type to load
#     scene_options = {
#         "InteractiveTraversableScene": "Procedurally generated scene with fully interactive objects",
#         # "StaticTraversableScene": "Monolithic scene mesh with no interactive objects",
#     }
#     scene_type = "InteractiveTraversableScene"

#     # Choose the scene model to load
#     scenes = get_available_og_scenes(
#     ) if scene_type == "InteractiveTraversableScene" else get_available_g_scenes()
#     # scene_model = choose_from_options(
#     #     options=scenes, name="scene model", random_selection=random_selection)
#     scene_model = "Beechwood_0_int"

#     cfg = {
#         "scene": {
#             "type": scene_type,
#             "scene_model": scene_model,
#         }
#     }

#     # If the scene type is interactive, also check if we want to quick load or full load the scene
#     if scene_type == "InteractiveTraversableScene":
#         load_options = {
#             "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
#             "Full": "Load all interactive objects in the scene",
#         }
#         load_mode = choose_from_options(
#             options=load_options, name="load mode", random_selection=random_selection)
#         if load_mode == "Quick":
#             cfg["scene"]["load_object_categories"] = [
#                 "floors", "walls", "ceilings"]

#     # Load the environment
#     env = og.Environment(configs=cfg)
#     # Always shut down the environment cleanly at the end
#     env.close()
