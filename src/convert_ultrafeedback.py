import os
import json

import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer

from tqdm import tqdm

from src.instruction import INSTRUCTION_TEXT, PROMPT_TEMPLATE


def get_stats(arr):
    return [
        min(arr), max(arr), np.mean(arr), np.percentile(arr, 25), np.percentile(arr, 50),
        np.percentile(arr, 75), np.percentile(arr, 95), np.percentile(arr, 99)
    ]


def convert_ultrafeedback(
    output_file=None,
    max_tokens=2720,
    max_prompt_tokens=512,
    max_resp_tokens=1024,
    order_augment=True
):
    np.random.seed(442)

    if output_file:
        output_dir = os.path.split(output_file)[0]
        os.makedirs(output_dir, exist_ok=True)

    hf_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train_dict = hf_dataset['train_prefs'].to_dict()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    dataset, dataset_aug = [], []

    data_stats = {
        "input": [],
        "prompt": [],
        "resp_a": [],
        "resp_b": []
    }
    for idx in tqdm(range(len(train_dict["prompt"]))):
        prompt_text = train_dict["prompt"][idx]
        resp_chosen_text = train_dict["chosen"][idx][-1]["content"]
        resp_rejected_text = train_dict["rejected"][idx][-1]["content"]
        score_chosen = train_dict["score_chosen"][idx]
        score_rejected = train_dict["score_rejected"][idx]
        
        output_text = None
        if score_chosen <= score_rejected + 0.5:
            output_text = "tie"
        elif score_chosen - 2 < score_rejected:
            continue

        if np.random.randint(0, 2) == 0:
            resp_a_text = resp_chosen_text
            resp_b_text = resp_rejected_text
            if output_text is None:
                output_text = "a"
        else:
            resp_a_text = resp_rejected_text
            resp_b_text = resp_chosen_text
            if output_text is None:
                output_text = "b"

        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
        encoded_resp_a = tokenizer.encode(resp_a_text, add_special_tokens=False)
        encoded_resp_b = tokenizer.encode(resp_b_text, add_special_tokens=False)
        encoded_input_text = encoded_prompt + encoded_resp_a + encoded_resp_b
        if len(encoded_input_text) > max_tokens:
            encoded_prompt = encoded_prompt[:max_prompt_tokens]
            encoded_resp_a = encoded_resp_a[:max_resp_tokens]
            encoded_resp_b = encoded_resp_b[:max_resp_tokens]
            encoded_input_text = encoded_prompt + encoded_resp_a + encoded_resp_b

        prompt = tokenizer.decode(encoded_prompt)
        resp_a = tokenizer.decode(encoded_resp_a)
        resp_b = tokenizer.decode(encoded_resp_b)
        input_text = PROMPT_TEMPLATE.format_map({
            "prompt": prompt, 
            "resp_a": resp_a,
            "resp_b": resp_b
        })

        dataset.append({
            "instruction": INSTRUCTION_TEXT,
            "input": input_text,
            "output": output_text
        })
        
        if order_augment:
            input_text_aug = PROMPT_TEMPLATE.format_map({
                "prompt": prompt, 
                "resp_a": resp_b,
                "resp_b": resp_a
            })
            output_text_aug = "tie"
            if output_text == "a":
                output_text_aug = "b"
            elif output_text == "b":
                output_text_aug = "a"

            dataset_aug.append({
                "instruction": INSTRUCTION_TEXT,
                "input": input_text_aug,
                "output": output_text_aug
            })

        data_stats["input"].append(len(encoded_input_text))
        data_stats["prompt"].append(len(encoded_prompt))
        data_stats["resp_a"].append(len(encoded_resp_a))
        data_stats["resp_b"].append(len(encoded_resp_b))

    if output_file:
        if order_augment:
            np.random.shuffle(dataset)
            np.random.shuffle(dataset_aug)
            dataset = dataset_aug + dataset

        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=4)

    print("Ultrafeedback dataset stats")
    for k, v in data_stats.items():
        print(k, get_stats(v))
    
    return dataset, dataset_aug


if __name__ == "__main__":
    convert_ultrafeedback(output_file="./data/instruction/ultrafeedback.json")