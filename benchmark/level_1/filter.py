# this file is not necessary for the task_config, but is used to help accelerate the process of task sampling

import os
# from omnigibson.macros import gm
from omnigibson.maps import SegmentationMap
from constants import *
import json

root_path = "../OmniGibson/omnigibson/data/og_dataset/"

def filter_rooms():
    room_dict = {}
    for scene in SCENES:
        scene_model = scene
        og_scenes_path = os.path.join(root_path, "scenes", scene_model)
        seg_map = SegmentationMap(scene_dir=og_scenes_path)
        available_rooms = list(seg_map.room_ins_name_to_ins_id.keys())
        room_dict[scene_model] = available_rooms

    rooms = ["corridor"]
    legal_rooms = []
    for room in rooms:
        for scene in SCENES:
            scene_rooms = room_dict[scene]
            for scene_room in scene_rooms:
                if room in scene_room:
                    legal_rooms.append(scene)
        print(f"Legal rooms for {room}: {legal_rooms}")

# filter objects to find the scenes that contain the object
def filter_objects(object_name=None):
    avail_scenes = []
    for scene in SCENES:
        scene_model = scene
        og_scenes_path = os.path.join(root_path, "scenes", scene_model)
        config_file = os.path.join(og_scenes_path, f"json/{scene}_best.json")
        scene_config = json.load(open(config_file, "r"))
        # print(scene_config["state"]["object_registry"].keys())
        if object_name:
            for key in scene_config["state"]["object_registry"].keys():
                if object_name in key:
                    print(f"Object {object_name} found in {scene_model}, key: {key}")
                    if scene_model not in avail_scenes:
                        avail_scenes.append(scene_model)
    print(f"Scenes with {object_name}: {avail_scenes}")

if __name__ == "__main__":
    filter_rooms()
    # filter_objects("box")
    # # filter_objects("sofa")