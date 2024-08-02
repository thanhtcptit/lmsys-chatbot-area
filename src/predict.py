import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from tqdm import tqdm

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


SUBMIT_KAGGLE = False
MULTI_GPU = True
DO_EVAL = True

MAX_TOKENS = 1440
MAX_PROMPT_TOKENS = 256
MAX_RESPONSE_TOKENS = 512
BATCH_SIZE = 2

TEMPLATE = "Given the prompt below, evaluate which response is superior or if both responses are equally good.\n<Prompt> {prompt}\n<Response #a>: {resp_a}\n<Response #b>: {resp_b}\n### Answer:"

if SUBMIT_KAGGLE:
    MODEL_PATH = "/kaggle/input/gemma2-9b-4bit-unsloth"
    ADAPTER_PATH = "/kaggle/input/gemma2-9b-4bit-lora-lmsys-checkpoint-3500"
    TRAIN_DATA_PATH = "/kaggle/input/lmsys-chatbot-arena/train.csv"
    TEST_DATA_PATH = "/kaggle/input/lmsys-chatbot-arena/test.csv"
else:
    MULTI_GPU = False
    MODEL_PATH = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    ADAPTER_PATH = "saves/lmsys-short/llama3-8b/lora_sft/checkpoint-1800"
    TRAIN_DATA_PATH = "./data/csv/train.csv"
    TEST_DATA_PATH = "./data/csv/test.csv"


def process_text(text):
    text = " ".join(eval(text, {"null": "None"}))
    if not text:
        return "None"
    return text


def tokenize(sample, tokenizer, prompt_template, max_tokens, max_prompt_tokens, max_resp_tokens):
    prompt_text = process_text(sample["prompt"])
    resp_a_text = process_text(sample["response_a"])
    resp_b_text = process_text(sample["response_b"])
    
    encoded_prompt = tokenizer.encode(prompt_text)
    encoded_resp_a = tokenizer.encode(resp_a_text)
    encoded_resp_b = tokenizer.encode(resp_b_text)
    encoded_input_text = encoded_prompt + encoded_resp_a + encoded_resp_b
    if len(encoded_input_text) > max_tokens:
        input_text = prompt_template.format_map({
            "prompt": tokenizer.decode(encoded_prompt[:max_prompt_tokens]), 
            "resp_a": tokenizer.decode(encoded_resp_a[:max_resp_tokens]),
            "resp_b": tokenizer.decode(encoded_resp_b[:max_resp_tokens])
        })
    else:
        input_text = prompt_template.format_map({
            "prompt": prompt_text, 
            "resp_a": resp_a_text,
            "resp_b": resp_b_text
        })

    label = -1
    if "winner_model_a" in sample:
        if sample["winner_model_a"] == 1:
            label = 0
        elif sample["winner_model_b"] == 1:
            label = 1
        else:
            label = 2
    
    input_text = tokenizer.apply_chat_template([{"role": "user", "content": input_text}], tokenize=False, add_generation_prompt=True)
    encoded_text = tokenizer(input_text)

    return {
        **encoded_text,
        "labels": torch.tensor(label)
    }


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, tokenizer, device, batch_size, do_eval=False):
    label_ids = [tokenizer(i, add_special_tokens=False)["input_ids"][0] for i in ['a', 'b', 'tie']]

    a_win, b_win, tie = [], [], []
    loss, acc = 0, 0

    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx: end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        input_last_idx = torch.sum(inputs["attention_mask"], dim=-1) - 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[torch.arange(outputs.logits.shape[0]), input_last_idx]
        logits = logits[:, label_ids]
        preds = torch.argmax(logits, dim=-1)

        if do_eval:
            labels = torch.tensor(tmp["labels"].to_list(), device=device)
            loss += nn.CrossEntropyLoss(reduction="sum")(logits, labels)
            acc += torch.sum(preds == labels)
        
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())

    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie

    if do_eval:
        loss = loss.item() / len(df)
        acc = acc.item() / len(df)
        print(loss, acc)
    return df


def main(model_path, adapter_path, train_data_path, test_data_pathh, prompt_template,
         max_tokens, max_prompt_tokens, max_resp_tokens, batch_size, do_eval=False, multi_gpu=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.environ.get("HF_TOKEN", ""))
    tokenizer.padding_side = "right"

    if do_eval:
        data_df = pd.read_csv(train_data_path)
        _, test_df = train_test_split(data_df, test_size=100, random_state=42)
    else:
        test_df = pd.read_csv(test_data_pathh)
    test_dataset = Dataset.from_pandas(test_df)
    tokenized_datasets = test_dataset.map(
        tokenize, 
        fn_kwargs={"tokenizer": tokenizer, "prompt_template": prompt_template, "max_tokens": max_tokens,
                   "max_prompt_tokens": max_prompt_tokens, "max_resp_tokens": max_resp_tokens},
    )

    encoded_test_df = tokenized_datasets.to_pandas()
    encoded_test_df["max_len"] = encoded_test_df["input_ids"].apply(len)
    encoded_test_df = encoded_test_df.sort_values("max_len", ascending=False)

    device_0 = torch.device("cuda:0")
    base_model_0 = AutoModelForCausalLM.from_pretrained(model_path, device=device_0)
    model_0 = PeftModel.from_pretrained(base_model_0, model_id=adapter_path).to(device_0) 
    model_0.eval()
    if multi_gpu:
        device_1 = torch.device("cuda:1")
        base_model_1 = AutoModelForCausalLM.from_pretrained(model_path, device=device_1)
        model_1 = PeftModel.from_pretrained(base_model_1, model_id=adapter_path).to(device_1) 
        model_1.eval()

        sub_1 = encoded_test_df.iloc[0::2].copy()
        sub_2 = encoded_test_df.iloc[1::2].copy()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(
                inference, (sub_1, sub_2), (model_0, model_1), (tokenizer, tokenizer),
                (device_0, device_1), (batch_size, batch_size))
        result_df = pd.concat(list(results), axis=0)

        proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values
        result_df.loc[:, "winner_model_a"] = proba[:, 0]
        result_df.loc[:, "winner_model_b"] = proba[:, 1]
        result_df.loc[:, "winner_tie"] = proba[:, 2]
        submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
    else:
        submission_df = inference(encoded_test_df, model_0, tokenize, device_0, batch_size, do_eval)

    submission_df.to_csv('submission.csv', index=False)
    return submission_df


submission_df = main(MODEL_PATH, ADAPTER_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, TEMPLATE,
                     MAX_TOKENS, MAX_PROMPT_TOKENS, MAX_RESPONSE_TOKENS, BATCH_SIZE, DO_EVAL, MULTI_GPU)
