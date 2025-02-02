from preference import Preference
from constants import *
import random
import os
import logging
import omnigibson as og
og.log.setLevel(logging.ERROR)
from omnigibson.objects import DatasetObject
from omnigibson.macros import gm
from omnigibson.maps import SegmentationMap
from omnigibson_modified import TraversableMap
from benchmark.pick_and_place import pick_and_place
import numpy as np
from utils import (
    get_trav_map,
    convert
)
from omnigibson.utils.asset_utils import get_og_scene_path
from omnigibson.robots import Fetch
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from omnigibson.sensors import VisionSensor
import matplotlib
from video_recorder import video_recorder
matplotlib.use('TkAgg')

def plot_trace(start, target, path, trav_map_size):
    # plot the path
    start = convert(start, trav_map_size)
    target = convert(target, trav_map_size)
    plt.scatter(start[0], start[1], c="r", s=trav_map_size)
    plt.scatter(target[0], target[1], c="b", s=trav_map_size)
    for i in range(len(path) - 1):
        pos = convert(path[i], trav_map_size)
        next_pos = convert(path[i + 1], trav_map_size)
        plt.plot([pos[0], next_pos[0]], [
                 pos[1], next_pos[1]], c="g", linewidth=5)
        
class Task:
    # we define task here
    # For rearrangement task, level 0 means home level, level 1 means detailed level
    def __init__(self, 
                task_flag: str = "Rearrangement",
                task_level: int = 0,
                task_name: str = "",
                scene_model: str = "Beechwood_0_int",
                **kwargs):
        self.task_flag = task_flag
        assert self.task_flag in ["Rearrangement", "SequencePlanning"]
        self.task_level = task_level
        self.task_name = task_name
        self.scene_model = scene_model
        self.check = True
        self.reset()

    def import_scene(self):
        self_path = os.path.dirname(os.path.abspath(__file__))
        config_filename = os.path.join(self_path, "../scene_config.yaml")
        cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        cfg["scene"]["scene_model"] = self.scene_model
        if og.sim:
            og.sim.clear()
        env = og.Environment(configs=cfg)
        self.scene = env.scene

    def import_robot(self):
        self.robot = og.sim.scene.object_registry("name", "robot_0")
        
        # get random pos in the scene
        pos = self.trav_map.get_random_point(floor=0)[1]
        self.robot.set_position(pos)
        
        # Expand the filed of view
        for sensor in self.robot.sensors.values():
            if isinstance(sensor, VisionSensor):
                sensor.horizontal_aperture = 50

    def import_objects(self, size=222):
        self.imported_objects = []
        for cfg in self.obj_cfgs:
            obj = DatasetObject(
                name=cfg["name"],
                category=cfg["category"],
                model=cfg["model"],
                # bounding_box=cfg["bounding_box"],
                fit_avg_dim_volume=True,
                position=convert(cfg["position"][1], size)
            )
            og.sim.import_object(obj)
            obj.set_position(cfg["position"][1])
            self.imported_objects.append(obj)

    def reset(self):
        self.trav_map_img, self.trav_map_size = get_trav_map(self.scene_model)
        self.trav_map = TraversableMap()
        self.trav_map.load_map(os.path.join(get_og_scene_path(
            self.scene_model), "layout"))
        self.trav_map.build_trav_graph(self.trav_map_size, os.path.join(get_og_scene_path(
            self.scene_model), "layout"), 1, self.trav_map_img.copy())
        # # for quick load
        # self.cfg["scene"]["load_object_categories"] = [
        #     "floors", "walls", "ceilings"]
        og_scenes_path = os.path.join(gm.DATASET_PATH, "scenes", self.scene_model)

        self.seg_map = SegmentationMap(scene_dir=og_scenes_path)
        self.available_rooms = list(self.seg_map.room_ins_name_to_ins_id.keys())
        self.preference = Preference(self.task_flag, self.task_level, self.task_name)
        # self.scene = InteractiveTraversableScene(self.scene_model)
        # self.scene = InteractiveTraversableScene(self.scene_model, \
            # load_object_categories=["walls", "floors", "countertop", "breakfast_table", "shelf"])
        self.import_scene()
        self.obj_cfgs = self.preference.set_object_cfgs(self.available_rooms, self.seg_map, self.trav_map, scene=self.scene)
        # print(self.obj_cfgs)
        if not self.obj_cfgs:
            self.check = False
            return
        # # self.scene = InteractiveTraversableScene("Rs_int")
        self.import_robot()
        self.import_objects(self.trav_map_size)

    def plot_objects(self):
        for cfg in self.obj_cfgs:
            pos = cfg["position"][1]
            pos = convert(pos, self.trav_map_size)
            plt.scatter(pos[0], pos[1], c="r", s=self.trav_map_size)
    
    def plot_obj_destination(self):
        for cfg in self.obj_cfgs:
            pos = cfg["target_pos"][1]
            pos = convert(pos, self.trav_map_size)
            plt.scatter(pos[0], pos[1], c="b", s=self.trav_map_size)

    def step(self):
        for i, obj in enumerate(self.imported_objects):
            target_pos = self.obj_cfgs[i]["target_pos"][1]
            target_recep = None if self.obj_cfgs[i]["target_recep"] is None \
                else og.sim.scene.object_registry("name", self.obj_cfgs[i]["target_recep"])
            og.log.info(f"robot pos: {self.robot.get_position()}, \
                  obj_pos: {obj.get_position()}, \
                  target: {target_pos}")
            will_on_floor = True if not target_recep else False
            pick_and_place(
                obj, 
                pre_recep=None, 
                is_on_floor=True, 
                is_in=False, 
                target_recep=target_recep, 
                target_pos=target_pos, 
                will_on_floor=will_on_floor, 
                robot=self.robot, 
                scene_model=self.scene_model, 
                trav_map=self.trav_map, 
                trav_map_img=self.trav_map_img, 
                trav_map_size=self.trav_map_size
            )
        # # Loop until the user requests an exit
        # exit_now = False
        # while not exit_now:
        #     og.sim.step()
            
    def init_figure(self, 
                    camera_pos = np.array([2.32248, -8.74338, 9.85436]),
                    camera_ori = np.array([0.39592, 0.13485, 0.29286, 0.85982]),
                    save_path=".", 
                    save_name="task_sample_desk_test"):
        # plt.figure(figsize=(12, 12))
        # plt.imshow(self.trav_map_img)
        # plt.title(f"Traversable area of {self.scene_model} scene")

        # Update the viewer camera's pose so that it points towards the robot
        og.sim.viewer_camera.set_position_orientation(position=camera_pos, orientation=camera_ori)
        video_recorder.set(camera=og.sim.viewer_camera, robot=self.robot, \
            save_path=os.path.join(f"{og.root_path}/../../dataset/", save_path), name=save_name,trav_map_img=self.trav_map_img)
        og.log.info(og.root_path)
        
    def save_figure(self):
        plt.savefig(f"{og.root_path}/../../images/pick_and_place_objects.png")
        return 
    
    def close(self):
        # Always shut the simulation down cleanly at the end
        # og.app.close()
        og.sim.stop()

if __name__ == "__main__":
    task = Task(task_flag="Rearrangement", task_level=0, task_name="food in kitchen", scene_model="Benevolence_1_int")
    task.init_figure(save_path=f"../images", 
                                 save_name=f"food in kitchen_Benevolence_1_int")
    task.step()
    task.close()