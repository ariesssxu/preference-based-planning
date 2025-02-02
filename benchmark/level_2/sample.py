from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.simulator import launch_simulator
import omnigibson as og
import itertools
import numpy as np
import random
from video_recorder import video_recorder
from constants import *
from level_0.task import Task

# Sample tasks for sequence planning in different scenes. You can customize the scenes and subtasks here. However, there may be some scenes that are not suitable for some subtasks.
SCENES = ["Beechwood_0_int", "Benevolence_1_int", "Ihlen_1_int", "Pomaria_1_int", "Wainscott_0_int", "house_double_floor_lower",
          "house_single_floor", "Merom_1_int", "restaurant_brunch", "restaurant_cafeteria", "Rs_garden", "Wainscott_0_garden"]
SUBTASKS = ["Cook", "Clean", "Wash", "Pour", "PickAndPlace"]


def sample_task(sub_task_num=3, sample_num=3, scenes=SCENES, subtasks=SUBTASKS):

    print("----------------------")
    print(f"Sample sequence tasks")
    print("----------------------")

    # sequence_preferences = list(itertools.permutations(subtasks, sub_task_num))
    # sequence_preferences = [["Cook", "Clean", "Wash", "Pour", "PickAndPlace"]]
    sequence_preferences = [["Cook"], ["Pour"],
                            ["PickAndPlace"], ["Clean"], ["Wash"]]
    for scene_model in scenes:
        scene = InteractiveTraversableScene(scene_model)
        og.sim.import_scene(scene)
        # if scene_model == "Benevolence_1_int":
        #     sequence_preferences = [["PickAndPlace"], ["Clean"], ["Wash"]]
        for sequence_preference in sequence_preferences:
            for i in range(sample_num):
                sub_task_nums = dict(
                    zip(sequence_preference, np.random.randint(1, 3, len(sequence_preference))))
                # sub_task_nums = dict(zip(sequence_preference, [1]))

                task = Task(task_flag="SequencePlanning", task_name=f"{sequence_preference}", task_id=i,
                            scene_model=scene_model, sub_task_nums=sub_task_nums)
                task.init_figure()
                task.step()
                task.close()

    # Always shut the simulation down cleanly at the end
    og.app.close()


if __name__ == "__main__":
    launch_simulator()
    sample_task(sub_task_num=3, sample_num=3,
                scenes=SceneInfo.keys(), subtasks=Subtasks.keys())
