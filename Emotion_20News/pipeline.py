from typing import List
import random
import copy
import os
import numpy as np
from datasets import load_dataset, Dataset
from torch import nn
# from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
print("Loaded examples.glue.pipeline")

# Copied from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py.
GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def construct_bert(data_name: str = "sst2") -> nn.Module:
    config = AutoConfig.from_pretrained(
        "bert-base-cased",
        num_labels=2,
        finetuning_task=data_name,
        trust_remote_code=True,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )


# 为 AG News 数据集定义 construct_bert 函数
def construct_bert_agnews() -> nn.Module:
    config = AutoConfig.from_pretrained(
        "bert-base-cased",
        num_labels=4,  # AG News 数据集有 4 个类别
        finetuning_task="agnews",
        trust_remote_code=True,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )

# 为 Emotion 数据集定义 construct_bert 函数
def construct_bert_emotion() -> nn.Module:
    config = AutoConfig.from_pretrained(
        "bert-base-cased",
        num_labels=6,  # Emotion 数据集有 6 个类别
        finetuning_task="emotion",
        trust_remote_code=True,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )


def construct_bert_20news() -> nn.Module:
    config = AutoConfig.from_pretrained(
        "bert-base-cased",
        num_labels=20,  # AG News 数据集有 4 个类别
        finetuning_task="20news",
        trust_remote_code=True,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )

def get_glue_dataset(
    data_name: str,
    split: str,
    indices: List[int] = None,
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    raw_datasets = load_dataset(
        path="glue",
        name=data_name,
    )
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    assert num_labels == 2

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, trust_remote_code=True)

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[data_name]
    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
    )

    if split in ["train", "eval_train"]:
        train_dataset = raw_datasets["train"]
        ds = train_dataset
        if data_name == "rte":
            ds = ds.select(range(2432))
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds



def get_ag_news_dataset(
    split: str,
    indices: List[int] = None,
) -> Dataset:
    """
    AG News 数据集包含 "train" 和 "test" 两个官方切分。
    可根据需要将 "train" 对应到 "train"/"eval_train"，把 "test" 对应到 "valid" 或 "test"。
    """
    assert split in ["train", "test", "valid"], \
        "AG News暂时与GLUE保持同样的split命名: train/test/valid"

    raw_datasets = load_dataset("ag_news")  # {'train': Dataset, 'test': Dataset}

    # AG News 为四个分类标签
    label_list = raw_datasets["train"].features["label"].names  # ['World', 'Sports', 'Business', 'Sci/Tech']
    print(f"[AG News] label_list = {label_list}")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, trust_remote_code=True)
    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        # AG News的文本字段通常是"text" 
        texts = examples["text"]
        result = tokenizer(
            texts,
            padding=padding,
            max_length=max_seq_length,
            truncation=True
        )
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    # 给 train 和 test 做同样的映射
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
    )

    # if split in ["train", "eval_train"]:
    #     ds = raw_datasets["train"]
    # else:
    #     ds = raw_datasets["test"]
    ds = raw_datasets[split]

    if indices is not None:
        ds = ds.select(indices)

    return ds








def get_emotion_dataset(
    split: str,
    indices: List[int] = None,
) -> Dataset:
    """
    Emotion 数据集包含 "train", "validation", "test" 三个切分，
    并且共有 6 个标签: 'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'.
    
    这里示例与GLUE格式保持一致:
      - split in ["train", "eval_train"] => 使用原始的 "train"
      - split in ["valid"] => 使用原始的 "validation"
    你也可自行修改, 比如专门处理 "test" 的情况等。
    """
    assert split in ["train", "test", "validation"], \
        "Emotion暂时与GLUE保持同样的split命名: train/test/validation"

    raw_datasets = load_dataset("emotion")  # {'train', 'validation', 'test'}

    label_list = raw_datasets["train"].features["label"].names
    print(f"[Emotion] label_list = {label_list}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, trust_remote_code=True)
    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        texts = examples["text"]
        result = tokenizer(
            texts,
            padding=padding,
            max_length=max_seq_length,
            truncation=True
        )
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
    )


    if split in ["train", "eval_train"]:
        noise_path = 'emotion_noise.npy'
        ds = raw_datasets["train"]

        if os.path.exists(noise_path):
            new_labels = np.load(noise_path).tolist()
        else:
            print('===============', type(ds), len(ds["labels"]), np.max(ds["labels"]), np.min(ds["labels"]))
            new_labels = copy.deepcopy(ds["labels"])
            noise_offset = 1
            noise_percent = 0.4
            for i in range(len(ds["labels"])):
                cur_rand = random.random()
                if cur_rand <= noise_percent:
                    new_labels[i] = (new_labels[i] + noise_offset) % 6
                    noise_offset += 1
                    if noise_offset % 6 == 0:
                        noise_offset = 1
            errors = np.abs(np.array(new_labels) - np.array(ds["labels"]))
            errors[errors > 0] = 1
            print(new_labels[:10])
            print(ds["labels"][:10])
            print(errors)
            print('!!!!!!!!!!Error Percent:', np.sum(errors), len(new_labels), np.sum(errors)/len(new_labels))
            np.save(noise_path, np.array(new_labels))

        data_dict = ds.to_dict()
        data_dict["labels"] = new_labels
        ds = Dataset.from_dict(data_dict)
    else:
        ds = raw_datasets["validation"]

    if indices is not None:
        ds = ds.select(indices)

    return ds














def get_20_news_dataset(
    split: str,
    indices: List[int] = None,
) -> Dataset:
    """
    20 News 数据集包含 "train" 和 "test" 两个官方切分。
    可根据需要将 "train" 对应到 "train"/"eval_train"，把 "test" 对应到 "valid" 或 "test"。
    """
    assert split in ["train", "test", "valid"], \
        "20 News暂时与GLUE保持同样的split命名: train/test/valid"

    raw_datasets = load_dataset("SetFit/20_newsgroups")  # {'train': Dataset, 'test': Dataset}   
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, trust_remote_code=True)
    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        # AG News的文本字段通常是"text" 
        texts = examples["text"]
        result = tokenizer(
            texts,
            padding=padding,
            max_length=max_seq_length,
            truncation=True
        )
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    # 给 train 和 test 做同样的映射
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
    )
    # ds = raw_datasets[split]


    if split in ["train", "eval_train"]:
        noise_path = '20news_noise.npy'
        ds = raw_datasets["train"]

        if os.path.exists(noise_path):
            new_labels = np.load(noise_path).tolist()
        else:
            print('===============', type(ds), len(ds["labels"]), np.max(ds["labels"]), np.min(ds["labels"]))
            num_classes = int(np.max(ds["labels"]) + 1)
            new_labels = copy.deepcopy(ds["labels"])
            noise_offset = 1
            noise_percent = 0.4
            for i in range(len(ds["labels"])):
                cur_rand = random.random()
                if cur_rand <= noise_percent:
                    new_labels[i] = (new_labels[i] + noise_offset) % num_classes
                    noise_offset += 1
                    if noise_offset % num_classes == 0:
                        noise_offset = 1
            errors = np.abs(np.array(new_labels) - np.array(ds["labels"]))
            errors[errors > 0] = 1
            print(new_labels[:10])
            print(ds["labels"][:10])
            print(errors)
            print('!!!!!!!!!!Error Percent:', np.sum(errors), len(new_labels), np.sum(errors)/len(new_labels))
            np.save(noise_path, np.array(new_labels))

        data_dict = ds.to_dict()
        data_dict["labels"] = new_labels
        ds = Dataset.from_dict(data_dict)
    else:
        ds = raw_datasets[split]

    if indices is not None:
        ds = ds.select(indices)

    return ds




















if __name__ == "__main__":
    from kronfluence import Analyzer

    model = construct_bert()
    print(Analyzer.get_module_summary(model))
