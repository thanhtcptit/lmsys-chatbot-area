import os
import json

import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from tqdm import tqdm

from src.instruction import INSTRUCTION_TEXT, PROMPT_TEMPLATE
from src.convert_ultrafeedback import convert_ultrafeedback


def get_stats(arr):
    return [
        min(arr), max(arr), np.mean(arr), np.percentile(arr, 25), np.percentile(arr, 50),
        np.percentile(arr, 75), np.percentile(arr, 95), np.percentile(arr, 99)
    ]


def process_text(text):
    text = " ".join(eval(text, {"null": "None"}))
    return text


def convert_lmsys(
    data_file,
    output_file,
    max_tokens=2720,
    max_prompt_tokens=512,
    max_resp_tokens=1024,
    order_augment=True,
    length_augment=True,
    use_ultrafeedback=False,
):
    np.random.seed(442)

    output_dir = os.path.split(output_file)[0]
    os.makedirs(output_dir, exist_ok=True)

    data_df = pd.read_csv(data_file)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    dataset, dataset_aug = [], []

    data_stats = {
        "input": [],
        "prompt": [],
        "resp_a": [],
        "resp_b": []
    }
    for idx, sample in tqdm(data_df.iterrows(), total=len(data_df)):
        prompt_text = process_text(sample["prompt"])
        resp_a_text = process_text(sample["response_a"])
        resp_b_text = process_text(sample["response_b"])

        if sample["winner_model_a"] == 1:
            output_text = "a"
        elif sample["winner_model_b"] == 1:
            output_text = "b"
        else:
            output_text = "tie"

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

        if length_augment:
            prompt = tokenizer.decode(encoded_prompt[:np.random.randint(256, max_prompt_tokens)])
            resp_a = tokenizer.decode(encoded_resp_a[:np.random.randint(512, max_resp_tokens)])
            resp_b = tokenizer.decode(encoded_resp_b[:np.random.randint(512, max_resp_tokens)])

        dataset.append({
            "instruction": INSTRUCTION_TEXT,
            "input": input_text,
            "output": output_text
        })

        if order_augment:
            if length_augment:
                prompt = tokenizer.decode(encoded_prompt[:np.random.randint(128, 256)])
                resp_a = tokenizer.decode(encoded_resp_a[:np.random.randint(352, 512)])
                resp_b = tokenizer.decode(encoded_resp_b[:np.random.randint(352, 512)])

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

    if use_ultrafeedback:
        ultrafeedback_dataset, ultrafeedback_dataset_aug = convert_ultrafeedback(order_augment=True)
        dataset += ultrafeedback_dataset
        dataset_aug += ultrafeedback_dataset_aug

    np.random.shuffle(dataset)
    np.random.shuffle(dataset_aug)
    dataset = dataset_aug + dataset
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    print("LMSYS dataset stats")
    for k, v in data_stats.items():
        print(k, get_stats(v))


if __name__ == "__main__":
    train_csv = "./data/csv/train.csv"
    output_file = "./data/instruction/lmsys.json"
    convert_lmsys(train_csv, output_file)
