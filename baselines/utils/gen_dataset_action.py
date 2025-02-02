# Scripts for generating dataset for action recognition

import json
import os
import re

id = 1
data_dir = f"~/PbP/dataset/sample_task_{id}"

action_categroies = []
data = []

list_dir = os.listdir(data_dir)
# for item in list_dir:
#     categroy = item.split("_")[0] if id == 1 else item
#     if categroy not in categroies:
#         categroies.append(categroy)

# to index
# categroy2index = {}
# for i, item in enumerate(categroies):
#     categroy2index[item] = i

for data_item in list_dir:
    if "json" in data_item:
        continue
    categroy = data_item.split("_")[0] if id == 1 else data_item
    scene = data_item.split("_")[1] if id == 1 else None
    if id == 1:
        for data_id in os.listdir(os.path.join(data_dir, data_item)):
            video_dir = os.path.join(data_dir, data_item, data_id, "")
            data_point = os.listdir(video_dir)
            for item in data_point:
                if "robot" in item:
                    camera_file = os.path.join(video_dir, item)
                elif "map" in item:
                    map_file = os.path.join(video_dir, item)
                elif "text" in item:
                    text_file = os.path.join(video_dir, item)
            with open(text_file, 'r') as f:
                actions = f.readlines()
                actions = [action.strip().split(" ")[0] for action in actions]
            action_categroies.extend(actions)

        print(list(set(action_categroies)))
                

            # data.append({
            #     "category": categroy2index[categroy],
            #     "preference": categroy,
            #     "scene": scene,
            #     "camera": camera_file,
            #     "map": map_file,
            #     "text": text_file
            # })
    elif id == 0:
        for scene in os.listdir(os.path.join(data_dir, data_item)):
            scene_dir = os.path.join(data_dir, data_item, scene, "")
            for data_id in os.listdir(scene_dir):
                video_dir = os.path.join(scene_dir, data_id, "")
                data_point = os.listdir(video_dir)
                print(data_point)
                for item in data_point:
                    if "robot" in item:
                        camera_file = os.path.join(video_dir, item)
                    elif "map" in item:
                        map_file = os.path.join(video_dir, item)
                    elif "text" in item:
                        text_file = os.path.join(video_dir, item)
                    # use re to filter all words and connect with "_"
                preference = re.findall(r'[A-Z][a-z]*', data_item)
                preference = "_".join(preference)
                print(preference)
                data.append({
                    "category": categroy2index[categroy],
                    "preference": preference,
                    "scene": scene,
                    "camera": camera_file,
                    "map": map_file,
                    "text": text_file
                })

# print(data[0])
# print("Category: ", len(categories))
# print("Len dataset:", len(data))

# with open(os.path.join(data_dir, "train_action.json"), "w") as f:
#     for item in data[:int(len(data) * 0.8)]:
#         json.dump(item, f)
#         f.write("\n")

# with open(os.path.join(data_dir, "test_action.json"), "w") as f:
#     for item in data[int(len(data) * 0.8):]:
#         json.dump(item, f)
#         f.write("\n")
        
# # print all preferences as a text
# preferences = []
# for item in categroies:
#     preference = re.findall(r'[A-Z][a-z]*', item)
#     preference = "_".join(preference)
#     preferences.append(preference)
# print(preferences)