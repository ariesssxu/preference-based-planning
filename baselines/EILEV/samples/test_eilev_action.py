import argparse
import logging
from pathlib import Path
import random
import torch
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import Blip2Processor

from eilev.data.utils import generate_input_ids_and_labels_from_interleaved
from eilev.model.utils import process
from eilev.model.v2 import VideoBlipForConditionalGeneration

import json
from tqdm.auto import tqdm
import sys
sys.path.append('../../benchmark/task_1/')
from constants import *
from utils import *

def generate(
    model: VideoBlipForConditionalGeneration,
    processor: Blip2Processor,
    videos_and_texts: list[str],
) -> None:
    video_path_handler = VideoPathHandler()
    # uniformly subsample 8 frames
    subsampler = UniformTemporalSubsample(8)
    prompts: list[tuple[str, int]] = [("", 0)]
    frames_list: list[torch.Tensor] = []
    for video_or_text in videos_and_texts:
        stripped = video_or_text.strip()
        if stripped.endswith(".mp4") or stripped.endswith(".avi"):
            # we have a video, so start a new text block
            # if the previous text block is not empty
            if prompts[-1][0] != "":
                prompts.append(("", 0))

            # process only the first 8 seconds if the video is longer than 8 seconds
            video = video_path_handler.video_from_path(stripped)
            end_sec = min(video.duration, 8)
            clip = video.get_clip(0, end_sec)

            frames = process(
                processor, video=subsampler(clip["video"].to(torch.uint8))
            ).pixel_values.squeeze(0)
            frames_list.append(frames)
            text_block, num_video = prompts[-1]
            prompts[-1] = (text_block, num_video + 1)
        else:
            logging.debug(f'"{stripped}" is not a file, so treating it as text.')
            text_block, num_video = prompts[-1]
            if text_block != "":
                text_block += " "
            text_block += stripped
            prompts[-1] = (text_block, num_video)
            
    inputs = generate_input_ids_and_labels_from_interleaved(
        processor.tokenizer,
        prompts,
        None,
        model.config.num_query_tokens,
        model.config.use_decoder_only_language_model,
    )

    # process the inputs
    generate_kwargs = {
        "pixel_values": torch.stack(frames_list).to(model.device),
        "input_ids": inputs["input_ids"].unsqueeze(0).to(model.device),
        "video_input_mask": inputs["video_input_mask"].unsqueeze(0).to(model.device),
        "max_new_tokens": 256,
        "num_beams": 5,
        "do_sample": False,
        "length_penalty": -1,
    }
    if model.config.text_config.architectures[0] == "OPTForCausalLM":
        # if the LLM is OPT, set eos_token_id to the newline character as this is the
        # setting used by BLIP-2.
        # https://github.com/salesforce/LAVIS/blob/7f00a0891b2890843f61c002a8e9532a40343648/lavis/models/blip2_models/blip2_opt.py#L91-L93
        generate_kwargs["eos_token_id"] = 50118

    generated_ids = model.generate(**generate_kwargs)  # type: ignore
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    # print(f"Generated_text: {generated_text}")
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate action narrations using an EILEV-trained model."
    )
    parser.add_argument("--model", default="kpyu/eilev-blip2-opt-2.7b")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model = VideoBlipForConditionalGeneration.from_pretrained(args.model).to(
        args.device
    )
    if args.processor is None:
        args.processor = args.model
    processor = Blip2Processor.from_pretrained(args.processor)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        
    data_dir = "../../dataset/sample_task_1/"
    train_file = f"{data_dir}train.json"

    with open(train_file, "r") as f:
        lines = f.readlines()
        test_files = [json.loads(line) for line in lines]
        random.shuffle(test_files)
        false = 0

        total_distance = 0

        for n, test_file in enumerate(tqdm(test_files[:-2])):

            with open(test_file["text"], 'r') as f:
                text_log = f.readlines()

            video_path_1 = test_files[n+1]["camera"]
            preference_1 = test_files[n+1]["preference"]
            with open(test_files[n+1]["text"], 'r') as f:
                text_log_1 = f.readlines()
            
            video_path_2 = test_files[n+2]["camera"]
            preference_2 = test_files[n+2]["preference"]
            with open(test_files[n+2]["text"], 'r') as f:
                text_log_2 = f.readlines()
            
            video_path_3, preference_3, text_log_3 = get_same_demo_video_text(test_file["preference"], test_files)
            with open(test_file["text"], 'r') as f:
                text_log = f.readlines()
            
            demos = [[video_path_1, preference_1, text_log_1], [video_path_2, preference_2, text_log_2], [video_path_3, preference_3, text_log_3]]
            random.shuffle(demos)
            video_path_1, preference_1, text_log_1 = demos[0]
            video_path_2, preference_2, text_log_2 = demos[1]
            video_path_3, preference_3, text_log_3 = demos[2]

            instructions = "You are a robot assistant. Please read the text log file and help generate action sequence."
            # possible_preferences = Rearrangement[0]['Level0'] + Rearrangement[2]['Level2']
            # possible_preferences = Sequence_Preferences['name']

            # instructions += "Quesiton: What's the user's preference? Choose from the preference listed before:"
            instruction_1 = f"{parse_concat(text_log_1, replace=', ')}.\n"
            instruction_2 = f"{parse_concat(text_log_2, replace=', ')}.\n"
            instruction_3 = f"{parse_concat(text_log_3, replace=', ')}.\n"
            instruction_4 = f"Now you have object A,B,C, Demo action sequence 4: \n"
            
            videos_and_texts = [instructions, video_path_1, instruction_1, video_path_2, instruction_2, video_path_3]
            # videos_and_texts = [instruction_1, instruction_2, video_path_1, preference_1, video_path_2, preference_2, video_path]
            # videos_and_texts = [instruction_1, instruction_2, video_path_1, preference_1, video_path_2, preference_2, video_path_3, preference_3, video_path]
            # videos_and_texts = [video_path, instruction_2]

            print(videos_and_texts)
    
            answer = generate(model, processor, videos_and_texts)

            print("answer:", answer)
            
            answer = answer.strip()

            gt = parse_concat(text_log, replace=', ')

            levenshtein_distance = compute_levenshtein_distance(answer, gt)
            print(f"Levenshtein distance: {levenshtein_distance}")

            total_distance += levenshtein_distance
            print(f"Step {n+1} average distance: {total_distance/(n+1)}")

        print(f"Average distance: {total_distance/len(test_files)}")

