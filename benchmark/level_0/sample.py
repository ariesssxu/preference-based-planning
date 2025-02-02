from task_2.task import Task
from constants import *
from video_recorder import video_recorder
import random
import numpy as np
import itertools
import omnigibson as og
from omnigibson.simulator import launch_simulator
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene

SCENES = ["Beechwood_0_int", "Benevolence_1_int", "Ihlen_1_int", "Pomaria_1_int", "Rs_int", "Wainscott_0_int"]
SUBTASKS = ["Cook", "Clean", "Wash", "Pour", "PickAndPlace"]

def sample_sequence_task(sub_task, sample_num=3, scenes=SCENES):

    print("----------------------")
    print(f"Sample task {sub_task}")
    print("----------------------")

    for scene_model in scenes:
        scene = InteractiveTraversableScene(scene_model, not_load_object_categories=["ceilings"])
        og.sim.import_scene(scene)
        for i in range(sample_num):            
            task = Task(task_flag="TaskPreference", task_name=f"{sub_task}", task_id=i,\
                                scene_model=scene_model)
            task.init_figure()
            task.step()
            task.close()
            video_recorder.release()
    
    # Always shut the simulation down cleanly at the end
    og.app.close()

if __name__ == "__main__":
    launch_simulator()
    sample_sequence_task(sub_task=SUBTASKS[0], sample_num=2, scenes=SceneInfo.keys())