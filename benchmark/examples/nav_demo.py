import time
import numpy as np
import os
import yaml
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_og_scenes, get_og_scene_path
from omnigibson.maps import SegmentationMap
from omnigibson_modified import TraversableMap
import cv2
import imageio
from omnigibson.objects import DatasetObject
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib
from utils import get_trav_map
matplotlib.use('TkAgg')

# Make sure object states, GPU dynamics, and transition rules are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = True


def convert(xy, size, resolution=0.1):
    return (xy / resolution + size / 2.0)


def get_robot_view(robot):
    robot_view = robot.get_obs()
    rgb_sensor = list(robot_view.keys())[0]
    obs = robot_view[rgb_sensor]
    return obs


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

    scene_model = "house_single_floor"

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
        "floors", "ceilings"]

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()

    og_dataset_path = gm.DATASET_PATH
    og_scenes_path = os.path.join(og_dataset_path, "scenes", scene_model)
    
    # test segmentation map
    # map = SegmentationMap(scene_dir=og_scenes_path)
    # print(map.get_room_type_by_point([0, 0]))

    trav_map_img, trav_map_size = get_trav_map(scene_model)

    if not headless:
        plt.figure(figsize=(12, 12))
        plt.imshow(trav_map_img)
        plt.title(f"Traversable area of {scene_model} scene")

    # if not headless:
    #     plt.show()

    # don't know why, but double loading is needed
    trav_map = TraversableMap()
    trav_map.load_map(os.path.join(get_og_scene_path(
        scene_model), "layout"))
    trav_map.build_trav_graph(trav_map_size, os.path.join(get_og_scene_path(
        scene_model), "layout"), 0, trav_map_img)

    # astar path example
    # start = trav_map.get_random_point(floor=None)[1][:-1]
    # target = trav_map.get_random_point(floor=None)[1][:-1]
    start = np.array([4.7,-3.])
    target = np.array([0.1,0.])
    print("Start point: ", start, "Target point: ", target)
    path = trav_map.get_shortest_path(
        floor=0, source_world=start, target_world=target, entire_path=True)[0]
    print("Astar navigation path: ", path)

    # draw path on the traversable map

    # Reset the robot
    robot = env.robots[0]
    robot.set_position([start[0],  start[1],  0])
    robot.reset()
    robot.keep_still()
    obs_video = []
    # get ego-centric view of the robot
    # obs_video.append(get_robot_view(robot))

    # video_writer = imageio.get_writer(f"{og.root_path}/../images/nav.mp4", fps=30)

    # Run a simple loop and reset periodically
    i = 0
    max_iterations = 1
    for j in range(max_iterations):
        og.log.info("Resetting environment")
        env.reset()
        for pos in path:
            robot.set_position([pos[0], pos[1], 0])
            # action = env.action_space.sample()
            # state, reward, done, info = env.step(action)
            # for _ in range(10):
            #     robot.set_position([pos[0], pos[1], 0])
            #     video_writer.append_data(get_robot_view(robot)[:, :, :-1])
            #     og.sim.step()

    # video_writer.close()

    # plot the path
    start = convert(start, trav_map_size)
    target = convert(target, trav_map_size)
    plt.scatter(start[0], start[1], c="r", s=200)
    plt.scatter(target[0], target[1], c="b", s=200)
    for i in range(len(path) - 1):
        pos = convert(path[i], trav_map_size)
        next_pos = convert(path[i + 1], trav_map_size)
        plt.plot([pos[0], next_pos[0]], [
                 pos[1], next_pos[1]], c="g", linewidth=10)

    # save the image
    plt.savefig("path.png")
    
    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()

