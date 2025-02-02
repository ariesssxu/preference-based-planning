import transformers
import torch
import json
import random
from tqdm.auto import tqdm
import sys
sys.path.append('../../benchmark/level_0/')
from constants import *
from utils import *

model_id = "./Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
  temperature = 0.05,
  max_new_tokens=30,
)


data_dir = "../../dataset/level_0"
train_file = f"{data_dir}/train.json"

with open(train_file, "r") as f:
    lines = f.readlines()
    test_files = [json.loads(line) for line in lines]
    random.shuffle(test_files)
    false = 0

    for n, test_file in enumerate(tqdm(test_files)):

        with open(test_file["text"], 'r') as f:
            text_log = f.readlines()

        instructions = "You are a robot assistant that can help summarize the host's preference. Please read the text log file and summarize the user's preference."
        # possible_preferences = Rearrangement[0]['Level0'] + Rearrangement[2]['Level2']
        possible_preferences = Sequence_Preferences['name']

        instructions += f"Choose from following preference: \n{parse_concat(possible_preferences, replace=', ')}.\n"
        instructions += f"Please summarize the preference from the text log file: \n {parse_concat(text_log, replace=', ')}.\n"
        instructions += "Quesiton: What's the user's preference? Choose from the preference listed before:"

        response = pipeline(f"{instructions}\nThe user's preference is ")

        answer = response[0]['generated_text'].split("The user's preference is ")[1].strip()
        
        print(answer)
        
        gt = test_file["preference"]

        answer = answer.lower()

        for keyword in gt.split(" ")[:]:
            if keyword.lower() not in answer:
                false += 1
                print(f"False: {answer} vs {gt}")
                break

        print(f"True: {n+1-false}/{n+1}")
        
    print(f"True: {n+1-false}/{len(test_files)}")