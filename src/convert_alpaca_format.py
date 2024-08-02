import os
import json

import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from tqdm import tqdm


LENGTH_CONFIGS = {
    "long": {
        "max_tokens": 2720,
        "max_prompt_tokens": 512,
        "max_resp_tokens": 1024
    },
    "extr": {
        "max_tokens": 3500,
        "max_prompt_tokens": 768,
        "max_resp_tokens": 1280
    }
}


def get_stats(arr):
    return [
        min(arr), max(arr), np.mean(arr), np.percentile(arr, 25), np.percentile(arr, 50),
        np.percentile(arr, 75), np.percentile(arr, 95), np.percentile(arr, 99)
    ]


def process_text(text, tokenizer=None, max_length=None):
    text = " ".join(eval(text, {"null": "None"}))
    if not text:
        return "None"
    if max_length and tokenizer:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            text = tokenizer.decode(tokens[:max_length])
    return text


def main():
    train_csv = "./data/csv/train.csv"
    output_file = "./data/instruction_alpaca/lmsys3-long.json"
    output_dir = os.path.split(output_file)[0]
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    max_tokens = 2720
    max_prompt_tokens = 512
    max_resp_tokens = 1024
    order_augment = False

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    instruction_text = "Given a prompt and two responses #a and #b, evaluate which response is superior or if both responses are equally good."
    # prompt_template = "<Prompt> {prompt}\n<Response #a>: {resp_a}\n<Response #b>: {resp_b}\n### Answer:"

    # prompt_template = "<Prompt>:<|reserved_special_token_50|>\n{prompt}\n<|reserved_special_token_51|>\n\n<Response #a>:\n<|reserved_special_token_52|>\n{resp_a}\n<|reserved_special_token_53|>\n\n<Response #b>:\n<|reserved_special_token_54|>\n{resp_b}\n<|reserved_special_token_55|>\n\nAnswer with a, b, or tie.\n### Answer:"
    
    prompt_template = "<|reserved_special_token_100|>\n<Prompt>:\n{prompt}\n<|reserved_special_token_110|>\n\n<|reserved_special_token_120|>\n<Response #a>:\n{resp_a}\n<|reserved_special_token_130|>\n\n<|reserved_special_token_140|>\n<Response #b>:\n{resp_b}\n<|reserved_special_token_150|>\n\nEvaluate which response is superior or if both responses are equally good. Answer with a, b, or tie.\n### Answer:"
    dataset, dataset_aug = [], []

    data_stats = {
        "input": [],
        "prompt": [],
        "resp_a": [],
        "resp_b": []
    }
    for idx, sample in tqdm(train_df.iterrows(), total=len(train_df)):
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

        input_text = prompt_template.format_map({
            "prompt": tokenizer.decode(encoded_prompt), 
            "resp_a": tokenizer.decode(encoded_resp_a),
            "resp_b": tokenizer.decode(encoded_resp_b)
        })

        dataset.append({
            "instruction": instruction_text,
            "input": input_text,
            "output": output_text
        })

        if order_augment:
            input_text_aug = prompt_template.format_map({
                "prompt": tokenizer.decode(encoded_prompt), 
                "resp_a": tokenizer.decode(encoded_resp_b),
                "resp_b": tokenizer.decode(encoded_resp_a)
            })
            output_text_aug = "tie"
            if output_text == "a":
                output_text_aug = "b"
            elif output_text == "b":
                output_text_aug = "a"

            dataset_aug.append({
                "instruction": instruction_text,
                "input": input_text_aug,
                "output": output_text_aug
            })

        data_stats["input"].append(len(encoded_input_text))
        data_stats["prompt"].append(len(encoded_prompt))
        data_stats["resp_a"].append(len(encoded_resp_a))
        data_stats["resp_b"].append(len(encoded_resp_b))

    if order_augment:
        np.random.shuffle(dataset_aug)
        dataset += dataset_aug

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    for k, v in data_stats.items():
        print(k, get_stats(v))


if __name__ == "__main__":
    main()