#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Union

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.utils import PaddingStrategy
import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="stanford_alpaca",
)

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT_CHINESE_GLM10B = {
    "prompt_input": (
        "以下是描述任务的说明，以及该任务的一个输入。编写一个响应以完成请求。\n\n"
        "### 提示：\n{instruction}\n\n### 输入：\n{input}\n\n### 回复：[gMASK]"
    ),
    "prompt_no_input": (
        "以下是描述任务的说明，编写一个响应以完成请求。\n\n"
        "### 提示：\n{instruction}\n\n### 回复：[gMASK]"
    ),
}

PROMPT_DICT_CHINESE = {
    "prompt_input": (
        "以下是描述任务的说明，以及该任务的一个输入。编写一个响应以完成请求。\n\n"
        "### 提示：\n{instruction}\n\n### 输入：\n{input}\n\n### 回复：[MASK]"
    ),
    "prompt_no_input": (
        "以下是描述任务的说明，编写一个响应以完成请求。\n\n"
        "### 提示：\n{instruction}\n\n### 回复：[MASK]"
    ),
}
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    # def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
    #     super(SupervisedDataset, self).__init__()
    #     logging.warning("Loading data...")
    #     list_data_dict = utils.jload(data_path)

    #     logging.warning("Formatting inputs...")
    #     prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    #     list_data_dict = [x for x in list_data_dict if ('instruction' in x and "input" in x and "output" in x)] # incase the data format is wrong, jing
    #     print(len(list_data_dict))
    #     sources = [
    #         prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    #         for example in list_data_dict
    #     ]
    #     targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

    #     logging.warning("Tokenizing inputs... This may take some time...")
    #     data_dict = preprocess(sources, targets, tokenizer)

    #     self.input_ids = data_dict["input_ids"]
    #     self.labels = data_dict["labels"]

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        if "glm-10b-chinese" in tokenizer.init_kwargs['name_or_path']:            
            prompt_input, prompt_no_input = PROMPT_DICT_CHINESE_GLM10B["prompt_input"], PROMPT_DICT_CHINESE_GLM10B["prompt_no_input"]
        else:
            prompt_input, prompt_no_input = PROMPT_DICT_CHINESE["prompt_input"], PROMPT_DICT_CHINESE["prompt_no_input"]
        list_data_dict = [x for x in list_data_dict if ('instruction' in x and "input" in x and "output" in x)] # incase the data format is wrong, jing
        print(len(list_data_dict))
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]
        self.sources = sources 
        self.targets = targets
        self.tokenizer = tokenizer
        # inputs = tokenizer(sources,return_tensors="pt", max_length=512, padding="max_length")
        # inputs = tokenizer.build_inputs_for_generation(inputs, targets=targets, max_gen_length=256, padding=False)
        # self.data_dict = inputs

    def __len__(self):
        return len(self.sources)
        # return len(self.data_dict['input_ids'])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(self.sources[i],return_tensors="pt", max_length=512, padding="max_length",truncation=True)
        inputs = self.tokenizer.build_inputs_for_generation(inputs, targets=self.targets[i], max_gen_length=256, padding="max_length")
        return inputs


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        batch = dict(
            input_ids = torch.cat([f['input_ids'] for f in features],dim = 0),
            labels = torch.cat([f['labels'] for f in features],dim = 0),
            position_ids = torch.cat([f['position_ids'] for f in features],dim = 0),
            attention_mask = torch.torch.cat([f['attention_mask'] for f in features],dim = 0)
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, trust_remote_code=True)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
