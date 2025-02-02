import transformers
import torch
import json
from tqdm.auto import tqdm
from constants import *
from utils import *
import random

model_id = "./Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
    temperature=0.05,
    max_new_tokens=30,
)


data_dir = "/data"
train_file = f"{data_dir}/train.json"

with open(train_file, "r") as f:
    lines = f.readlines()
    test_files = [json.loads(line) for line in lines]

    random.shuffle(test_files)

    false = 0

    for n, test_file in enumerate(tqdm(test_files[:-2])):

        # Naive sampling code. You can replace it with your own sampling code to test different combinations.
        with open(test_file["text"], 'r') as f:
            text_log = f.readlines()

        with open(test_files[n+1]["text"], 'r') as f:
            text_log_1 = f.readlines()
        preference_1 = test_files[n+1]["preference"]

        with open(test_files[n+2]["text"], 'r') as f:
            text_log_2 = f.readlines()
        preference_2 = test_files[n+2]["preference"]

        text_log_3, preference_3 = get_same_demo(
            test_file["preference"], test_files)

        # shuffle the text logs and corresponding preferences
        text_logs = [[text_log_1, preference_1], [
            text_log_2, preference_2], [text_log_3, preference_3]]
        random.shuffle(text_logs)
        text_log_1, preference_1 = text_logs[0]
        text_log_2, preference_2 = text_logs[1]
        text_log_3, preference_3 = text_logs[2]

        instructions = "You are a robot assistant that can help summarize the host's preference."
        # possible_preferences = Rearrangement[0]['Level0'] + Rearrangement[2]['Level2']
        possible_preferences = Sequence_Preferences['name']

        instructions += f"Choose from following preference: \n{parse_concat(possible_preferences, replace=', ')}.\n"
        # instructions += "Quesiton: What's the user's preference? Choose from the preference listed before:"
        instructions += f"Text log file: \n {parse_concat(text_log_1, replace=', ')}.\n"
        instructions += f"Preference: {preference_1}.\n"
        instructions += f"Text log file: \n {parse_concat(text_log_2, replace=', ')}.\n"
        instructions += f"Preference: {preference_2}.\n"
        instructions += f"Text log file: \n {parse_concat(text_log_3, replace=', ')}.\n"
        instructions += f"Preference: {preference_3}.\n"
        instructions += f"Text log file: \n {parse_concat(text_log, replace=', ')}.\n"
        instructions += f"The user's preference is "

        response = pipeline(f"{instructions}")

        answer = response[0]['generated_text'].split(
            "The user's preference is ")[1].strip()

        # print(answer)

        gt = test_file["preference"]

        answer = answer.lower()

        if not compare(answer, gt, in_sequence=True):
            print(f"{answer} != {gt}")
            false += 1

        print(f"True: {n+1-false}/{n+1}")

    print(f"True: {n+1-false}/{len(test_files)}")
