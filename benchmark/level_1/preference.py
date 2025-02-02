# We define preference here
from constants import *
from utils import (
    set_object_by_room, 
    get_semantic_map,
    convert_obj_cfg,
    get_random_point_by_recap
)
import numpy as np
import random
import omnigibson as og

def naive_config_sample(
            selected_obj_types, 
            object_config_dict,
            available_rooms,
            segmentation_map,
            tra_map, 
            each_obj_num=1
        ):
    obj_configs = {}
    for obj_type in selected_obj_types:
        # random sample num_objects obj_type in object_config_dict:
        obj_configs[obj_type] = []
        for i in range(each_obj_num):
            obj_configs[obj_type].append(
                set_object_by_room(
                    obj_type, 
                    object_config_dict, 
                    available_rooms,
                    segmentation_map,
                    tra_map, 
                    i
                )
            )
    return obj_configs

    
def set_objects_to_same_room(
            selected_obj_types, 
            object_config_dict,
            available_rooms,
            segmentation_map,
            tra_map,
            obj_num=3,
            target_room=None
        ):
    if target_room:
        filtered_target_rooms = [room for room in available_rooms if target_room in room]
        # check if target room in avail rooms
        if len(filtered_target_rooms) == 0:
            og.log.error("No available_room in current scene.")
            og.log.info(available_rooms)
            return False
            assert 0
    else:
        filtered_target_rooms = available_rooms
    available_rooms = list(set(available_rooms) - set(filtered_target_rooms))
    obj_configs = naive_config_sample(
        selected_obj_types, 
        object_config_dict,
        available_rooms,
        segmentation_map,
        tra_map,
    )
    target_room = np.random.choice(filtered_target_rooms)
    # target_room = np.random.choice(available_rooms)
     
    for obj_type in obj_configs:
        for config in obj_configs[obj_type]:
            config["target_room"] = target_room
            config["target_pos"] = segmentation_map.get_random_point_by_room_instance(target_room)
            config["target_recep"] = None
            
    # z are all set to zero here
    return obj_configs

def set_objects_to_common_room(
            selected_obj_types, 
            object_config_dict,
            available_rooms,
            segmentation_map,
            tra_map,
            obj_num=3,
            target_room=None
        ):
    obj_configs = naive_config_sample(
        selected_obj_types, 
        object_config_dict,
        available_rooms,
        segmentation_map,
        tra_map,
    )
    for obj_type in obj_configs:
        common_rooms = list(object_config_dict[obj_type]["room"].keys())
        target_room = random.choice(common_rooms)
        for config in obj_configs[obj_type]:
            config["target_room"] = target_room
            config["target_pos"] = segmentation_map.get_random_point_by_room_type(target_room)
            config["target_recep"] = None
    # z are all set to zero here
    return obj_configs

# tbd
def set_objects_to_same_object(
            selected_obj_types, 
            object_config_dict,
            available_rooms,
            segmentation_map,
            tra_map,
            target_recep=None,
        ):
    obj_configs = naive_config_sample(
            selected_obj_types, 
            object_config_dict,
            available_rooms,
            segmentation_map,
            tra_map,
        )
    for obj_type in obj_configs:
        # common_rooms = list(object_config_dict[obj_type]["room"].keys())
        # target_room = random.choice(common_rooms)
        for config in obj_configs[obj_type]:
            config["target_room"] = None
            config["target_pos"] = (0, get_random_point_by_recap(target_recep, bias=None))
            config["target_recep"] = target_recep.name
            # print("target obj:", target_recep.name, "pos: ", config["target_pos"])
    # z are all set to zero here
    return obj_configs

class Preference:
    def __init__(self, 
                    task_flag: str = "Rearrangement",
                    task_level: int = 0, 
                    task_name: str = ""):
        self.task_flag = task_flag
        self.task_level = task_level
        self.task_name = task_name
        self.rules = None
        if self.task_name != "":
            self.obj_type, _, self.target_place = self.task_name.split(" ")
        else:
            self.obj_type = None
            self.target_place = None

    def __str__(self):
        return f'{self.task_level}'

    def select_objects(self, single_type=False, obj_type=None):
        # For each task, we need to select 3 differet objects
        # This should based on self.preference, but for now we just randomly select
        self.object_config_dict = {}
        keys = ["food", "tools", "kitchenware", "toys", "clothes"]
        obj_type = random.choice(keys) if not obj_type else obj_type
        for obj, info in Objects[obj_type].items():
            self.object_config_dict[obj] = info
        
        # choose n objects from the object_config_dict
        obj_type_num = 1 if single_type else 3
        og.log.info(f"Objects: {list(self.object_config_dict.keys())}")
        self.selected_obj_types = random.sample(list(self.object_config_dict.keys()), obj_type_num)

    def set_object_cfgs(self, available_rooms, seg_map, trav_map, scene=None, single_type=False):
        self.select_objects(single_type, self.obj_type)
        # For each task, we need to set rules for each object
        obj_cfgs = self.sample_obj_cfgs(
            self.selected_obj_types, 
            self.object_config_dict,
            available_rooms,
            seg_map,
            trav_map,
            scene=scene,
        )
        return convert_obj_cfg(obj_cfgs)
    
    def select_target_recep(self, scene, name=None):
        avail_target_receps = []
        # for obj in scene.objects:
        #     # grep all tables, boxes, beds and shelfs
        #     if any([cate in obj.name for cate in ["countertop", "breakfast_table", "shelf", "bed", "fridge", ""]]):
        #         avail_target_receps.append(obj.name)
        if name is not None:
            avail_target_receps = [obj.name for obj in scene.objects if name in obj.name]
        target_recep_name = random.choice(avail_target_receps)
        obj = scene.object_registry("name", target_recep_name)
        return obj
    
    def sample_obj_cfgs(self, 
            selected_obj_types, 
            object_config_dict,
            available_rooms,
            segmentation_map,
            tra_map,
            single_type=False,
            scene=None,
        ):
        # # demo case: sample randomly
        # obj_configs = naive_config_sample(
        #     selected_obj_types, 
        #     object_config_dict,
        #     available_rooms,
        #     segmentation_map,
        #     tra_map,
        #     obj_configs,
        #     single_type
        # )
        og.log.info(f"{self.task_flag} task_level {self.task_level}")
        if self.task_name != "":
            og.log.info(f"Task {self.task_name}")
        # assert(0)
        if self.task_level == 0:
            # case 1: all objects to the same room
            obj_configs = set_objects_to_same_room(
                selected_obj_types, 
                object_config_dict,
                available_rooms,
                segmentation_map,
                tra_map,
                obj_num=3,
                target_room = self.target_place
            )
        
        elif self.task_level == 2:
            # case 3: object level
            target_recep = self.select_target_recep(scene, self.target_place)
            og.log.info(f"target_recep: {target_recep.name}")
            obj_configs = set_objects_to_same_object(
                selected_obj_types, 
                object_config_dict,
                available_rooms,
                segmentation_map,
                tra_map,
                target_recep,
            )
        return obj_configs
        