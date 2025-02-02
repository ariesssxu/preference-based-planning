import numpy as np
import os
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_og_scenes, get_og_scene_path
from omnigibson.maps import SegmentationMap, TraversableMap
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from utils import (
    sample_single_object, 
    convert, 
    get_robot_view, 
    get_semantic_map,
    set_object_config_list
)
from task_config.constants import *
matplotlib.use('TkAgg')

# Make sure object states, GPU dynamics, and transition rules are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = True

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

    # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()

    og_scenes_path = os.path.join(gm.DATASET_PATH, "scenes", scene_model)

    # test segmentation map
    segmentation_map = SegmentationMap(scene_dir=og_scenes_path)
    print(segmentation_map.get_room_instance_by_point([0, 0]))
    available_rooms = list(segmentation_map.room_ins_name_to_ins_id.keys())
    print(segmentation_map.get_random_point_by_room_type("bathroom"))
    img_ins, img_sem = get_semantic_map(scene_model)

    if not headless:
        plt.figure(figsize=(12, 12))
        plt.imshow(img_ins)
        plt.title(f"Semantic view area of {scene_model} scene")

    # get config file
    object_config_dict = {}
    for obj in Objects:
        for key, value in obj.items():
            object_config_dict[key] = list(value.keys())
    print(object_config_dict)

    current_room_configs = {}
    # choose n objects from the object_config_dict
    selected_obj_type = random.sample(list(object_config_dict.keys()), 3)
    # random sample 3 different objects:
    for obj_type in selected_obj_type:
        obj_type = np.random.choice(list(object_config_dict.keys()))
        current_room_configs[obj_type] = set_object_config_list(
            obj_type, 
            object_config_dict,
            available_rooms,
            segmentation_map,
            num_objects=1,
        )
    print(current_room_configs)

    # random sample 5 points in the bathroom
    # for i in range(5):
    #     point = segmentation_map.get_random_point_by_room_type("living_room")[1][:-1]
    #     point_2D = convert(point)
    #     # plot the point
    #     start = convert(point_2D)
    #     plt.scatter(start[0], start[1], c="r", s=200)

    # if not headless:
    #     plt.show()

    # plt.savefig("semantic_map_3.png")

    print(current_room_configs)
    # import object to the scene
    for obj_type in current_room_configs:
        for config in current_room_configs[obj_type]:
            print(config)
            # env.scene.add_object(**config)
            # or call import_object()?
        pos = current_room_configs[obj_type][0]["start_pos"]
        
    # # Load the environment
    # env = og.Environment(configs=cfg)

    # # Always close the environment at the end
    # env.close()


if __name__ == "__main__":
    main()
