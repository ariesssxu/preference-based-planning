# Description: This file contains the utility functions for the test scripts.
import random
import os
# check if gm is imported
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_all_object_category_models,
)
from PIL import Image
import numpy as np
import cv2
from math import atan2

from pyquaternion import Quaternion
from transforms3d.euler import euler2quat
import numpy as np
import matplotlib.pyplot as plt

def convert(xy, size=222, resolution=0.1):
    return np.array(xy) / resolution + size / 2.0

def get_robot_view(robot):
    robot_view = robot.get_obs()
    rgb_sensor = list(robot_view.keys())[0]
    obs = robot_view[rgb_sensor]
    return obs

def path_interpolation(path_list, steps=10):
    new_path_list = []
    for i in range(len(path_list) - 1):
        start = path_list[i]
        end = path_list[i + 1]
        for i in range(steps):
            new_path_list.append(start + (end - start) * i / steps)
    return new_path_list

def get_semantic_map(scene_model):
    og_dataset_path = gm.DATASET_PATH
    og_scenes_path = os.path.join(og_dataset_path, "scenes", scene_model)
    layout_dir = os.path.join(og_scenes_path, "layout")
    room_ins_imgs = os.path.join(layout_dir, "floor_insseg_0.png")
    img_ins = Image.open(room_ins_imgs)
    room_seg_imgs = os.path.join(layout_dir, "floor_semseg_0.png")
    img_sem = Image.open(room_seg_imgs)
    return img_ins, img_sem

def get_trav_map(scene_model, convert=True):
    og_dataset_path = gm.DATASET_PATH
    og_scenes_path = os.path.join(og_dataset_path, "scenes", scene_model)
    trav_map_erosion = 2
    trav_map_img = Image.open(os.path.join(og_scenes_path, "layout", "floor_trav_0.png"))
    if not convert:
        return trav_map_img, trav_map_img.size[0]
    trav_map_size = int(trav_map_img.size[0] * 0.1)
    trav_map_img = np.array(trav_map_img.resize((trav_map_size, trav_map_size)))
    trav_map_img = cv2.erode(trav_map_img, np.ones((trav_map_erosion, trav_map_erosion)))
    # print("trav_map_size: ", trav_map_size)
    return trav_map_img, trav_map_size

def set_object_by_room(
        obj_type, 
        object_config_dict,
        available_rooms,
        segmentation_map,
        tra_map,
        i=0
    ):
    category = obj_type
    vaild_room = object_config_dict[obj_type]
    start_room = np.random.choice(available_rooms)
    # start_pos = segmentation_map.get_random_point_by_room_type(start_room[:-2])
    start_pos = tra_map.get_random_point(floor=0)
    config = {
        "category": category, 
        "vaild_room": vaild_room, 
        "start_room": start_room, 
        "start_pos": start_pos,
        "id": i
    }
    return config

def convert_obj_cfg(ori_config_dict):
    if not ori_config_dict:
        return False
    avg_category_spec = get_og_avg_category_specs()
    obj_cfgs = []
    for obj_type in ori_config_dict:
        configs = ori_config_dict[obj_type]
        for config in configs:
            obj_path = os.path.join(gm.DATASET_PATH, "objects", config["category"])

            name = f"{config['category']}_{config['id']}"
            # Load the specs of the object categories, e.g., common scaling factor
            if "model" not in config:
                model = random.choice(os.listdir(obj_path))
                config["model"] = model
            # if "bounding_box" not in config:
            #     bounding_box = avg_category_spec.get(config["category"])
            #     config["bounding_box"] = bounding_box
            config["type"] = "DatasetObject"
            config["name"] = name
            config["position"] = config["start_pos"]
            config["fit_avg_dim_volume"] = True
            obj_cfgs.append(config)
    return obj_cfgs

# should add more constraints, e.g., detect collision or if the object is on the desk
def set_obj_position(obj):
    center_offset = obj.get_position() - obj.aabb_center + np.array([0, 0, obj.aabb_extent[2] / 2.0])
    obj.set_position(center_offset)

def get_random_point_by_recap(obj, bias=None):
    bounding_box = obj.native_bbox
    center_pos = obj.get_position()
    if obj.name.split("_")[0] in ["countertop", "bed", "table", "sofa"]:
        if not bias:
            # get random point on the contertop
            x = random.uniform(center_pos[0] - bounding_box[0] / 2.0, center_pos[0] + bounding_box[0] / 2.0)
            y = random.uniform(center_pos[1] - bounding_box[1] / 2.0, center_pos[1] + bounding_box[1] / 2.0)
        elif bias=="MIDDLE":
            x = random.uniform(center_pos[0] - bounding_box[0] / 4.0, center_pos[0] + bounding_box[0]  / 4.0)
            y = random.uniform(center_pos[1] - bounding_box[1] / 4.0, center_pos[1] + bounding_box[1] / 4.0)
        elif bias=="EDGE":
            x = center_pos[0] + random.choice([-1, 1]) * random.uniform(bounding_box[0] / 4.0, bounding_box[0] / 2.0)
            y = center_pos[1] + random.choice([-1, 1]) * random.uniform(bounding_box[1] / 4.0, bounding_box[1] / 2.0)
        else:
            print(f"Current bias {bias} not implemented for {obj.name}")
            raise NotImplementedError
        z = center_pos[2] + bounding_box[2] / 2.0
    elif obj.name.split("_")[0] in ["fridge", "shelf", "microwave", "sink"]:
        if not bias:
            # get random point on the contertop
            x = random.uniform(center_pos[0] - bounding_box[0] / 2.0, center_pos[0] + bounding_box[0] / 2.0)
            y = random.uniform(center_pos[1] - bounding_box[1] / 2.0, center_pos[1] + bounding_box[1] / 2.0)
            z = center_pos[2]
        else:
            print(f"Current bias {bias} not implemented for {obj.name}")
            raise NotImplementedError
    else:
        print(f"Current recap {obj.name} not implemented")
        raise NotImplementedError
    return np.array([x, y, z])


def robot_rotate_interpolation(start, end, steps=10):
    # start: [0, 0, a, b]
    # end: [0, 0, c, d]
    # rotate from a to c, b to d
    # return: [[0, 0, a, b], [0, 0, a + (c - a) * steps, b + (d - b) * steps], ...]
    # assert len(start) == 4
    # assert len(end) == 4
    # result = []
    # a_increment = (end[2] - start[2]) / steps
    # b_increment = (end[3] - start[3]) / steps

    # # Generate interpolated points
    # for i in range(steps + 1):
    #     a = start[2] + (a_increment * i)
    #     b = start[3] + (b_increment * i)
    #     result.append(np.array([0, 0, a, b]))
    # result.append(end)
    # return result
    result = []
    angle_increment = (end - start) / steps
    for i in range(steps + 1):
        angle = start + angle_increment * i
        quar = euler2quat(0, 0, angle)
        orientation = quar[[1, 2, 3, 0]]
        result.append(orientation)
    quar = euler2quat(0, 0, end)
    orientation = quar[[1, 2, 3, 0]]
    result.append(orientation)
    return result

def camera_rotate_interpolation(angles, start, end, steps=10):
    result = []
    angle_increment = (end - start) / steps
    for i in range(steps + 1):
        angle = start + angle_increment * i
        quar = euler2quat(angle, angles[1], angles[2])
        orientation = quar[[1, 2, 3, 0]]
        result.append(orientation)
    quar = euler2quat(end, angles[1], angles[2])
    orientation = quar[[1, 2, 3, 0]]
    result.append(orientation)
    return result

# get position functions
def get_robot_view(robot):
    robot_view = robot.get_obs()
    obs = robot_view['robot0:eyes_Camera_sensor_rgb']
    return obs

def available_pos_check(pos, trav_map, trav_map_size):
    pos = np.array(pos)
    pos = np.array([pos[0], pos[1]])
    pos = convert(pos, trav_map_size).astype(int)
    if pos[0] < 0 or pos[0] >= trav_map_size or pos[1] < 0 or pos[1] >= trav_map_size:
        return False
    if trav_map[pos[0], pos[1]] == 0:
        return False
    return True

def get_robot_position(obj, trav_map, trav_map_size):
    obj_pos, obj_ori = obj.get_position_orientation()
    bbox = obj.native_bbox
    if "openable" in obj.abilities:
        vec_standard = [[0, -1, 0]]
        if ((bbox[0] * 4) > (bbox[1] * 3)):
            vec_standard.append([0.8, -0.6, 0])
            vec_standard.append([-0.8, 0.6, 0])
            if ((bbox[0] * 3) > (bbox[1] * 4)):
                vec_standard.append([0.6, -0.8, 0])
                vec_standard.append([-0.6, 0.8, 0])
    else:
        vec_standard = [[0, -1, 0], [0.6, -0.8, 0], [0.8, -0.6, 0], [1, 0, 0], [0.8, 0.6, 0], [0.6, 0.8, 0], [0, 1, 0], [-0.6, 0.8, 0], [-0.8, 0.6, 0], [-1, 0, 0], [-0.8, -0.6, 0], [-0.6, -0.8, 0]]
    default_pos = np.zeros(3)
    for vec in vec_standard:
        rotated_vec = Quaternion(obj_ori[[3, 0, 1, 2]]).rotate(vec)
        if vec[0] == 0:
            distance = bbox[1]
        else:
            if vec[1] == 0:
                distance = bbox[0]
            else:
                if ((bbox[0] * abs(vec[1])) > (bbox[1] * abs(vec[0]))):
                    distance = bbox[1] / abs(vec[1])
                else:
                    distance = bbox[0] / abs(vec[0])
        robot_pos = np.zeros(3)
        robot_pos[0] = obj_pos[0] + rotated_vec[0] * distance * 0.5
        robot_pos[1] = obj_pos[1] + rotated_vec[1] * distance * 0.5
        robot_pos[2] = 0.005
        if obj.category == "sink":
            robot_pos[0] += rotated_vec[0] * 0.775
            robot_pos[1] += rotated_vec[1] * 0.775
        else:
            if "openable" in obj.abilities:
                robot_pos[0] += rotated_vec[0] * 0.6
                robot_pos[1] += rotated_vec[1] * 0.6
            else:
                robot_pos[0] += rotated_vec[0] * 0.5
                robot_pos[1] += rotated_vec[1] * 0.5
        # print("checking robot pos: ", vec)
        if(vec[1] == -1):
            default_pos = robot_pos
        if trav_map.has_node(0, np.array([robot_pos[0],robot_pos[1]])):
        # if available_pos_check(robot_pos, trav_map, trav_map_size):
            return robot_pos
    return default_pos

def get_robot_orientation(obj_pos, robot_pos):
    direction = np.zeros(2)
    direction[0] = obj_pos[0] - robot_pos[0]
    direction[1] = obj_pos[1] - robot_pos[1]
    angle = atan2(direction[1], direction[0])
    quar = euler2quat(0, 0, angle)
    
    return quar[[1, 2, 3, 0]]

def get_camera_orientation(angles, obj_pos, camera_pos):
    direction = np.zeros(2)
    direction[0] = ((obj_pos[0] - camera_pos[0]) ** 2 + (obj_pos[1] - camera_pos[1]) ** 2) ** 0.5
    direction[1] = obj_pos[2] - camera_pos[2]
    angle = atan2(direction[1], direction[0])
    quar =  euler2quat(angle + np.pi/2, angles[1], angles[2])

    return quar[[1, 2, 3, 0]]

def get_robot_angle(obj_pos, robot_pos):
    direction = np.zeros(2)
    direction[0] = obj_pos[0] - robot_pos[0]
    direction[1] = obj_pos[1] - robot_pos[1]
    angle = atan2(direction[1], direction[0])

    return angle

def get_camera_angle(obj_pos, camera_pos):
    direction = np.zeros(2)
    direction[0] = ((obj_pos[0] - camera_pos[0]) ** 2 + (obj_pos[1] - camera_pos[1]) ** 2) ** 0.5
    direction[1] = obj_pos[2] - camera_pos[2]
    angle = atan2(direction[1], direction[0])

    return angle + np.pi/2

def get_arm_position(robot):
    robot_pos, robot_ori = robot.get_position_orientation()
    vec_standard = np.array([1, 0, 0])
    rotated_vec = Quaternion(robot_ori[[3, 0, 1, 2]]).rotate(vec_standard)

    arm_pos = np.zeros(3)
    arm_pos[0] = robot_pos[0] + rotated_vec[0] * 0.3
    arm_pos[1] = robot_pos[1] + rotated_vec[1] * 0.3
    arm_pos[2] = 0.8
    
    return arm_pos

def get_operation_position(obj, dis=0.01):
    obj_pos, obj_ori = obj.get_position_orientation()
    vec_standard = np.array([0, -1, 0])
    rotated_vec = Quaternion(obj_ori[[3, 0, 1, 2]]).rotate(vec_standard)
    bbox = obj.native_bbox

    arm_pos = np.zeros(3)
    arm_pos[0] = obj_pos[0] + rotated_vec[0] * bbox[1] * 0.5 + rotated_vec[0] * dis
    arm_pos[1] = obj_pos[1] + rotated_vec[1] * bbox[1] * 0.5 + rotated_vec[1] * dis
    arm_pos[2] = obj_pos[2]

    return arm_pos