import json
import os
import re

id = 0
data_dir = f"~/PbP/dataset/sample_task_{id}"

categroies = []
data = []

list_dir = os.listdir(data_dir)
for item in list_dir:
    categroy = item.split("_")[0] if id == 1 else item
    if categroy not in categroies:
        categroies.append(categroy)

# to index
action_list = {'Cook', 'Clean', 'Wash', 'Pour', 'PickAndPlace'}
categroy_count = {}
for i, item in enumerate(action_list):
    categroy_count[item] = 0

for data_item in list_dir:
    categroy = data_item.split("_")[0] if id == 1 else data_item
    for a in action_list:
        if a in categroy:
            categroy_count[a] += 1

print(categroy_count)