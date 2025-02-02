from constants import *
from task import Task

# just sample 0 and 1 for example
print("Level 0 tasks: ", Rearrangement[0]['Level0'])
print("Level 1 tasks: ", Rearrangement[1]['Level1'])
print("All scenes: ", SCENES)

def sample_task(sample_num=5, scenes=SCENES, level_index=0):

    print("----------------------")
    print(f"Sample level {level_index} tasks")
    print("----------------------")
    
    for task_name in Rearrangement[level_index][f'Level{level_index}'][11:]:
        available_scenes = Scene_Config["Rearrange"][task_name] if task_name in Scene_Config["Rearrange"] else scenes
        available_scenes = list(set(available_scenes))
        if task_name == "kitchenware in sink":
            available_scenes = available_scenes[5:11]
        print(f"Available scenes for task {task_name}: {available_scenes}")
        for scene_model in available_scenes[:10]:
            if scene_model in ["Merom_1_int", "Wainscott_1_int"]:
                continue
            for i in range(sample_num):
                
                print("----------------------")
                print(f"Sample {task_name} in scene {scene_model} {i}th time")
                print("----------------------")
                
                try:
                    task = Task(task_flag="Rearrangement", task_level=level_index, task_name=task_name, scene_model=scene_model)
                    if not task.check:
                        print(f"Task {task_name} in scene {scene_model} is not available")
                        continue
                    task.init_figure(save_path=f"sample_task_1/{task_name}_{scene_model}/{i}", 
                                    save_name=f"{task_name}_{scene_model}_{i}")
                    task.step()
                    task.close()
                except Exception as e:
                    print(e)
                    continue

if __name__ == "__main__":
    sample_task(sample_num=10, level_index=2, scenes=SCENES[:])