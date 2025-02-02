import os
import random
import numpy as np
import omnigibson as og
from omnigibson_modified import TraversableMap
from omnigibson.utils.asset_utils import get_og_scene_path
from omnigibson.robots import Fetch
from omnigibson.sensors import VisionSensor
from omnigibson.macros import gm
gm.USE_GPU_DYNAMICS = True

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
                 task_flag: str = "SequencePlanning",
                 task_name: str = "",
                 task_id: int = 0,
                 scene_model: str = "Beechwood_0_int",
                 sub_task_nums: dict = {},
                 **kwargs):
        self.task_flag = task_flag
        assert self.task_flag in ["Rearrangement", "SequencePlanning"]
        self.task_name = task_name
        self.task_id = task_id
        self.scene_model = scene_model
        self.sub_task_nums = sub_task_nums
        self.scene = og.sim._scene
        self.sub_task_list = {}
        self.plate = None
        self.mug = None
        self.scene_objs = {}
        self.robot = None
        self.check = True
        self.reset()
    
    def import_scene(self):
    #     # self.scene = InteractiveTraversableScene(self.scene_model, not_load_object_categories=["ceilings"])
    #     # og.sim.import_scene(self.scene)
    #     # self.scene = og.sim._scene

    #     for obj_name in SceneObjects[self.scene_model]:
    #         obj = import_A_of_B(obj_name, SceneObjects[self.scene_model][obj_name]["model"])
    #         set_A_at_P(obj, SceneObjects[self.scene_model][obj_name]["pos"])
        available_surfaces = set(SceneInfo[self.scene_model]["surface"])
        for sub_task in self.sub_task_nums:
            if sub_task == "Cook":
                surface_name = random.choice(list(available_surfaces))
                available_surfaces = available_surfaces - set(surface_name)
                self.scene_objs["plate"] = surface_name
                surface = og.sim.scene.object_registry("name", surface_name)
                self.plate = import_A_of_B("plate", "itoeew")
                set_A_at_P(self.plate, surface.get_position() + np.array([0, 0, 0.5 * surface.native_bbox[2]]) + np.array([0, 0, 0.5 * self.plate.native_bbox[2]]))
            elif sub_task == "Pour":
                surface_name = random.choice(list(available_surfaces))
                available_surfaces = available_surfaces - set(surface_name)
                self.scene_objs["mug"] = surface_name
                surface = og.sim.scene.object_registry("name", surface_name)
                self.mug = import_A_of_B("mug", "fapsrj")
                set_A_at_P(self.mug, surface.get_position() + np.array([0, 0, 0.5 * surface.native_bbox[2]]) + np.array([0, 0, 0.5 * self.mug.native_bbox[2]]))

    def import_objects(self):
        loaded_obj = set()
        available_containers = set(SceneInfo[self.scene_model]["container"]) - set(self.scene_objs.values())
        available_funitures = set(SceneInfo[self.scene_model]["funiture"]) - set(self.scene_objs.values())
        
        for sub_task in self.sub_task_nums:
            self.sub_task_list[sub_task] = {}
            if sub_task == "Clean":
                obj_names = random.sample(available_funitures, self.sub_task_nums[sub_task])
            else:
                if (sub_task == "Cook") or (sub_task == "Pour"):
                    self.sub_task_nums[sub_task] = 1
                available_objs = set()
                for obj_type in Subtasks[sub_task]:
                    available_objs = available_objs | ObjectInfo[obj_type].keys()
                available_objs = available_objs - loaded_obj
                obj_names = random.sample(available_objs, self.sub_task_nums[sub_task])
            
            for obj_name in obj_names:
                loaded_obj.add(obj_name)
                print(sub_task, obj_name)
                if sub_task == 'Cook':
                    obj = import_A_of_B(obj_name)
                    while True:
                        obj_pos = self.trav_map.get_random_point(floor=0)[1]
                        set_A_at_P(obj, obj_pos + np.array([0, 0, 0.5 * obj.native_bbox[2]]))
                        robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                        if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                            break
                    self.sub_task_list[sub_task][obj_name] = None
                if sub_task == 'Wash':
                    obj = import_A_of_B(obj_name)
                    while True:
                        obj_pos = self.trav_map.get_random_point(floor=0)[1]
                        set_A_at_P(obj, obj_pos + np.array([0, 0, 0.5 * obj.native_bbox[2]]))
                        robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                        if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                            break
                    system = random.choice(["dust", "stain"])
                    import_A_on_B(system, obj)
                    self.sub_task_list[sub_task][obj_name] = system
                if sub_task == 'Clean':
                    obj = og.sim.scene.object_registry("name", obj_name)
                    system = random.choice(["dust", "stain"])
                    import_A_on_B(system, obj)
                    self.sub_task_list[sub_task][obj.name] = system
                if sub_task == 'Pour':
                    obj = import_A_of_B(obj_name)
                    while True:
                        obj_pos = self.trav_map.get_random_point(floor=0)[1]
                        set_A_at_P(obj, obj_pos + np.array([0, 0, 0.5 * obj.native_bbox[2]]))
                        robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                        if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                            break
                    system = random.choice(ObjectInfo["bottles"][obj_name]["contain"])
                    import_A_in_B(system, obj)
                    self.sub_task_list[sub_task][obj_name] = system
                if sub_task == 'PickAndPlace':
                    obj = import_A_of_B(obj_name)
                    while True:
                        obj_pos = self.trav_map.get_random_point(floor=0)[1]
                        set_A_at_P(obj, obj_pos + np.array([0, 0, 0.5 * obj.native_bbox[2]]))
                        robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                        if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                            break
                    container = random.choice(list(available_containers))
                    self.sub_task_list[sub_task][obj_name] = container

    def import_robot(self):
        self.robot = Fetch(
            prim_path="/World/robot",
            name="robot",
            fixed_base=True,
            scale=1.0,
            self_collisions=False,
            default_arm_pose="diagonal30",
            default_reset_mode="tuck",
            controller_config={
                "arm_0": {
                    "name": "NullJointController",
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
                sensor.horizontal_aperture = 50
                sensor.image_height = 512
                sensor.image_width = 512

    def remove_objects(self):
        robot = og.sim.scene.object_registry("name", "robot")
        if robot:
            og.sim.remove_object(robot)

        if self.plate:
            og.sim.remove_object(self.plate)
        if self.mug:
            og.sim.remove_object(self.mug)

        for sub_task in self.sub_task_list:
            for obj_name in self.sub_task_list[sub_task]:
                if sub_task != 'Clean':
                    obj = og.sim.scene.object_registry("name", obj_name)
                    og.sim.remove_object(obj)
                if (sub_task == 'Wash') or (sub_task == 'Clean') or (sub_task == 'Pour'):
                    system = get_system(self.sub_task_list[sub_task][obj_name])
                    system.clear()
                    if (sub_task == 'Wash') or (sub_task == 'Clean'):
                        system = get_system("water")
                        system.clear()
    
    def reset(self):
        self.trav_map_img, self.trav_map_size = get_trav_map(self.scene_model)
        self.trav_map = TraversableMap()
        self.trav_map.load_map(os.path.join(get_og_scene_path(
            self.scene_model), "layout"))
        self.trav_map.build_trav_graph(self.trav_map_size, os.path.join(get_og_scene_path(
            self.scene_model), "layout"), 1, self.trav_map_img.copy())
        
        og.sim.stop()
        self.remove_objects()
        
        self.import_robot()
        self.scene.reset()
        self.import_scene()
        self.import_objects()
        
        # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()
    
    def step(self):
        for sub_task in self.sub_task_list:
            for obj_name in self.sub_task_list[sub_task]:
                obj = og.sim.scene.object_registry("name", obj_name)
                if sub_task in ["Cook", "Wash", "Pour", "PickAndPlace"]:
                    while True:
                        robot_pos = get_robot_position(obj, self.trav_map, self.trav_map_size)
                        if self.trav_map.has_node(0, np.array([robot_pos[0], robot_pos[1]])):
                            break
                        obj_pos = self.trav_map.get_random_point(floor=0)[1]
                        set_A_at_P(obj, obj_pos + np.array([0, 0, 0.5 * obj.native_bbox[2]]))
                if sub_task == 'Cook':
                    cooker = og.sim.scene.object_registry("name", random.choice(SceneInfo[self.scene_model]["cooker"]))
                    surface = og.sim.scene.object_registry("name", self.scene_objs["plate"])
                    cook(obj, None, True, False, cooker, self.plate, surface, self.robot, self.scene_model,\
                         self.trav_map, self.trav_map_img, self.trav_map_size)
                if sub_task == 'Wash':
                    system = og.sim.scene.system_registry("name", self.sub_task_list[sub_task][obj_name])
                    sink = og.sim.scene.object_registry("name", random.choice(SceneInfo[self.scene_model]["sink"]))
                    wash(system, obj, None, True, False, sink, self.robot, self.scene_model,\
                         self.trav_map, self.trav_map_img, self.trav_map_size)
                if sub_task == 'Clean':
                    system = og.sim.scene.system_registry("name", self.sub_task_list[sub_task][obj_name])
                    clean(system, obj, self.robot, self.scene_model,\
                          self.trav_map, self.trav_map_img, self.trav_map_size)
                if sub_task == 'Pour':
                    system = og.sim.scene.system_registry("name", self.sub_task_list[sub_task][obj_name])
                    surface = og.sim.scene.object_registry("name", self.scene_objs["mug"])
                    pour(system, obj, None, True, False, self.mug, surface, self.robot, self.scene_model,\
                         self.trav_map, self.trav_map_img, self.trav_map_size)
                if sub_task == 'PickAndPlace':
                    container = og.sim.scene.object_registry("name", self.sub_task_list[sub_task][obj_name])
                    target_pos = container.get_position()
                    if container.category in ["breakfast_table","coffee_table", "shelf", "countertop", "desk"]:
                        target_pos = container.get_position()
                        target_pos[2] += 0.5 * container.native_bbox[2] + 0.5 * obj.native_bbox[2]
                    elif container.category in ["bottom_cabinet", "bottom_cabinet_no_top"]:
                        target_pos = get_operation_position(container, 0.25 * container.native_bbox[1])
                        target_pos[2] += 0.5 * container.native_bbox[2]
                    pick_and_place(obj, None, True, False, container, target_pos, False, self.robot, self.scene_model,\
                                   self.trav_map, self.trav_map_img, self.trav_map_size)

    def init_figure(self, 
                    camera_pos = np.array([2.32248, -8.74338, 9.85436]),
                    camera_ori = np.array([0.39592, 0.13485, 0.29286, 0.85982])):
        # plt.figure(figsize=(12, 12))
        # plt.imshow(self.trav_map_img)
        # plt.title(f"Traversable area of {self.scene_model} scene")

        # Update the viewer camera's pose so that it points towards the robot
        og.sim.viewer_camera.set_position_orientation(position=camera_pos, orientation=camera_ori)
        video_recorder.set(camera=og.sim.viewer_camera, robot=self.robot, save_path=os.path.join(f"{og.root_path}/../../dataset/",\
            f"{self.task_name}", f"{self.scene_model}", f"{self.task_id}"), name=f"{self.task_id}", trav_map_img=self.trav_map_img, trav_map_size=self.trav_map_size)
        og.log.info(og.root_path)
        
    # def save_figure(self):
    #     plt.savefig(f"{og.root_path}/../../images/sequence/{self.task_name}.png")
    #     return 
    
    def close(self):
        # video_recorder.release()
        og.sim.stop()
        self.remove_objects()