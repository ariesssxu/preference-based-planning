import os
import random
import numpy as np
import omnigibson as og
from omnigibson_modified import TraversableMap
from omnigibson.utils.asset_utils import get_og_scene_path
from omnigibson.robots import Fetch
from omnigibson.sensors import VisionSensor

from utils import (
    get_trav_map,
    get_robot_position
)
from constants import *
from video_recorder import video_recorder
from action import *
from sub_task import cook, pour, wash, clean, pick_and_place

class Task:
    # define sequence task here
    def __init__(self,
                 task_flag: str = "TaskPreference",
                 task_name: str = "",
                 task_id: int = 0,
                 scene_model: str = "Beechwood_0_int",
                 **kwargs):
        self.task_flag = task_flag
        assert self.task_flag in ["Rearrangement", "SequencePlanning", "TaskPreference"]
        self.sub_task = task_name
        self.task_id = task_id
        self.scene_model = scene_model
        self.scene = og.sim._scene
        self.robot = None
        self.task_list = {}
        self.check = True
        self.reset()
    
    def import_scene(self):
        # self.scene = InteractiveTraversableScene(self.scene_model, not_load_object_categories=["ceilings"])
        # og.sim.import_scene(self.scene)
        # self.scene = og.sim._scene

        for obj_name in SceneObjects[self.scene_model]:
            obj = import_A_of_B(obj_name, SceneObjects[self.scene_model][obj_name]["model"])
            set_A_at_P(obj, SceneObjects[self.scene_model][obj_name]["pos"])

    def import_objects(self):
        loaded_obj = set()
        if self.sub_task == "Clean":
            obj_name = random.choice(SceneInfo[self.scene_model]["funiture"])
        else:
            available_objs = set()
            for obj_type in Subtasks[self.sub_task]:
                available_objs = available_objs | ObjectInfo[obj_type].keys()
            available_objs = available_objs - loaded_obj
            obj_name = random.choice(list(available_objs))
        
        loaded_obj.add(obj_name)
        self.obj_name = obj_name
        print(self.sub_task, obj_name)
        if self.sub_task == 'Cook':
            obj = import_A_of_B(obj_name)
            while True:
                obj_pos = self.trav_map.get_random_point(floor=0)[1]
                set_A_at_P(obj, obj_pos)
                robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                    break
            self.task_list[obj_name] = None
        if self.sub_task == 'Wash':
            obj = import_A_of_B(obj_name)
            while True:
                obj_pos = self.trav_map.get_random_point(floor=0)[1]
                set_A_at_P(obj, obj_pos)
                robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                    break
            system = random.choice(["dust", "stain"])
            import_A_on_B(system, obj)
            self.task_list[obj_name] = system
        if self.sub_task == 'Clean':
            obj = og.sim.scene.object_registry("name", obj_name)
            system = random.choice(["dust", "stain"])
            import_A_on_B(system, obj)
            self.task_list[obj_name] = system
        if self.sub_task == 'Pour':
            obj = import_A_of_B(obj_name)
            while True:
                obj_pos = self.trav_map.get_random_point(floor=0)[1]
                set_A_at_P(obj, obj_pos)
                robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                    break
            system = random.choice(ObjectInfo["bottles"][obj_name]["contain"])
            import_A_in_B(system, obj)
            self.task_list[obj_name] = system
        if self.sub_task == 'PickAndPlace':
            obj = import_A_of_B(obj_name)
            while True:
                obj_pos = self.trav_map.get_random_point(floor=0)[1]
                set_A_at_P(obj, obj_pos)
                robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                    break
            container = random.choice(SceneInfo[self.scene_model]["container"])
            self.task_list[obj_name] = container

    def import_robot(self):
        self.robot = Fetch(
            prim_path="/World/robot",
            name="robot",
            fixed_base=True,
            controller_config={
                "arm_0": {
                    "name": "JointController",
                    "motor_type": "position"
                },
                "gripper_0": {
                    "name": "MultiFingerGripperController",
                    "mode": "binary"
                },
                "camera": {
                    "name": "JointController"
                }
            },
            grasping_mode="sticky",
            obs_modalities=["rgb"]
        )
        og.sim.import_object(self.robot)
        # get random pos in the scene

        pos = self.trav_map.get_random_point(floor=0)[1]
        self.robot.set_position(pos)
        
        # At least one simulation step while the simulator is playing must occur for the robot (or in general, any object)
        # to be fully initialized after it is imported into the simulator
        og.sim.play()
        og.sim.step()
        # Make sure none of the joints are moving
        self.robot.keep_still()
        # Expand the filed of view
        for sensor in self.robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                sensor.horizontal_aperture = 60
                sensor.image_height = 256
                sensor.image_width = 256

    def remove_objects(self):
        robot = og.sim.scene.object_registry("name", "robot")
        if robot:
            og.sim.remove_object(robot)

        for obj_name in SceneObjects[self.scene_model]:
            obj = og.sim.scene.object_registry("name", obj_name)
            if obj:
                og.sim.remove_object(obj)

        for obj_name in self.task_list:
            if self.sub_task != 'Clean':
                obj = og.sim.scene.object_registry("name", obj_name)
                og.sim.remove_object(obj)
            if (self.sub_task == 'Wash') or (self.sub_task == 'Clean') or (self.sub_task == 'Pour'):
                system = og.sim.scene.system_registry("name", self.task_list[obj_name])
                system.clear()
    
    def reset(self):
        og.sim.stop()
        self.remove_objects()
        og.sim.play()
        self.scene.reset()

        self.trav_map_img, self.trav_map_size = get_trav_map(self.scene_model)
        self.trav_map = TraversableMap()
        self.trav_map.load_map(os.path.join(get_og_scene_path(
            self.scene_model), "layout"))
        self.trav_map.build_trav_graph(self.trav_map_size, os.path.join(get_og_scene_path(
            self.scene_model), "layout"), 1, self.trav_map_img.copy())
        
        # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()
        
        self.import_scene()
        self.import_robot()
        self.import_objects()
    
    def step(self):
        obj = og.sim.scene.object_registry("name", self.obj_name)
        if self.sub_task in ["Cook", "Wash", "Pour", "PickAndPlace"]:
            while True:
                robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                    break
                obj_pos = self.trav_map.get_random_point(floor=0)[1]
                set_A_at_P(obj, obj_pos)
        if self.sub_task == 'Cook':
            cooker = og.sim.scene.object_registry("name", random.choice(SceneInfo[self.scene_model]["cooker"]))
            bowl = og.sim.scene.object_registry("name", "plate")
            cook(obj, None, True, False, cooker, bowl, self.robot, self.scene_model,\
                    self.trav_map, self.trav_map_img, self.trav_map_size)
        if self.sub_task == 'Wash':
            system = og.sim.scene.system_registry("name", self.obj_name)
            sink = og.sim.scene.object_registry("name", random.choice(SceneInfo[self.scene_model]["sink"]))
            wash(system, obj, None, True, False, sink, self.robot, self.scene_model,\
                    self.trav_map, self.trav_map_img, self.trav_map_size)
        if self.sub_task == 'Clean':
            system = og.sim.scene.system_registry("name", self.obj_name)
            clean(system, obj, self.robot, self.scene_model,\
                    self.trav_map, self.trav_map_img, self.trav_map_size)
        if self.sub_task == 'Pour':
            system = og.sim.scene.system_registry("name", self.obj_name)
            mug = og.sim.scene.object_registry("name", "mug")
            pour(system, obj, None, True, False, mug, self.robot, self.scene_model,\
                    self.trav_map, self.trav_map_img, self.trav_map_size)
        if self.sub_task == 'PickAndPlace':
            container = og.sim.scene.object_registry("name", self.obj_name)
            if container.category in ["armchair", "coffee_table", "shelf", "countertop", "desk"]:
                target_pos = container.get_position()
                target_pos[2] += 0.5 * container.native_bbox[2] + 0.5 * obj.native_bbox[2]
            elif container.category in ["bottom_cabinet", "bottom_cabinet_no_top"]:
                target_pos = get_operation_position(container)
                target_pos[2] += 0.5 * container.native_bbox[2]
            pick_and_place(obj, None, True, False, container, container.get_position(), False, self.robot, self.scene_model,\
                            self.trav_map, self.trav_map_img, self.trav_map_size)

    def init_figure(self, 
                    camera_pos = np.array([2.32248, -8.74338, 9.85436]),
                    camera_ori = np.array([0.39592, 0.13485, 0.29286, 0.85982])):

        # Update the viewer camera's pose so that it points towards the robot
        og.sim.viewer_camera.set_position_orientation(position=camera_pos, orientation=camera_ori)
        video_recorder.set(camera=og.sim.viewer_camera, robot=self.robot,\
                           save_path=os.path.join(f"{og.root_path}/../../dataset/sample_task_2", f"{self.sub_task}", f"{self.scene_model}", f"{self.task_id}"), name=f"{self.task_id}", trav_map_img=self.trav_map_img)
        og.log.info(og.root_path) 
    
    def close(self):
        # video_recorder.release()
        og.sim.stop()
        self.remove_objects()