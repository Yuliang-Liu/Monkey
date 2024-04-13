# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from monkey_model.modeling_textmonkey import TextMonkeyLMHeadModel
from monkey_model.tokenization_qwen import QWenTokenizer
from monkey_model.configuration_monkey import MonkeyConfig
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_llm: bool = False
    fix_resampler: bool = False
    image_size: int = 448
    image_width: int = 896
    image_height: int = 896
    n_queries: int = 256
    lora_repeat_num : int = 0
    add_window: bool  = False
    use_global: bool = True
    resampler: bool = False
    remain:int = 512
    


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["in_proj","out_proj","c_fc"] ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)



def format_tokenizer(tokenizer, message, return_target=False, label=False):
    _input_ids = tokenizer(message).input_ids
    input_ids =  _input_ids 
    if return_target:
        if label:
            target = input_ids
        else:
            target =  [IGNORE_TOKEN_ID] * (len(_input_ids)) 
        return input_ids, target
    else:
        return input_ids

def preprocess(
               source,
               tokenizer,
               max_len,
               system_message: str = "You are a helpful assistant.",
               padding=True
               ):
    '''
    [{"from": "user", "value": f"<img>{file_abs_path}</img>" + prefix,}, {"from": "assistant", "value": label}]

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <img>image_path<imgpad><imgpad><imgpad><imgpad> ... </img>Describe the image concisely.<|im_end|>
    <|im_start|>assistant
    A man on a surfboard on a wave in the ocean.<|im_end|>
    '''

    # Apply prompt templates
    input_ids, targets = [], []


    message_l = []
    for conv in source:
        message_l.append(conv["value"])
    for i, message in enumerate(message_l):
        try:
            _input_ids, _target = format_tokenizer(tokenizer, message, return_target=True, label=True if i %2==1 else False)  # <img> 有些text会有img标签，所以使用<img>作为特殊id有问题，标签数量不对等会报错
        except Exception as e:
            print(e)
            continue

        input_ids += _input_ids
        targets += _target 
        if i%2==1:
            input_ids += [-1]
            targets += [tokenizer.pad_token_id]
        assert len(_input_ids) == len(_input_ids)
    if padding:

        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))

        targets +=  [IGNORE_TOKEN_ID] * (max_len - len(targets))
        targets = targets[:max_len]
        input_ids = input_ids[:max_len]
        

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    attention_mask=input_ids.ne(tokenizer.pad_token_id)
    input_ids[input_ids == -1 ] = tokenizer.pad_token_id
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.raw_data[i]["conversations"], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"],
            labels=ret["labels"],
            attention_mask=ret["attention_mask"],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)
    
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def print_trainable_params(model: torch.nn.Module):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    rank0_print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param))
    for name,p in model.named_parameters():
        if p.requires_grad and "transformer.h" not in name and "lora" not in name:
            if "lora" in name:
                if "39" not in name:
                    continue
            rank0_print(name)
    # for name,p in model.named_parameters():
    #     rank0_print(name,p.device)

def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = MonkeyConfig.from_pretrained(
        "monkey_model",
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    config.visual["image_size"]= (training_args.image_height,training_args.image_width)
    config.visual["n_queries"]= training_args.n_queries
    config.visual["lora_repeat_num"]= training_args.lora_repeat_num
    config.visual["add_window"]= training_args.add_window
    config.visual["use_global"]= training_args.use_global
    config.visual["resampler"]= training_args.resampler
    config.visual["r"]= training_args.remain
    rank0_print(config)
    config.use_cache = False

    # Load model and tokenizer
    rank0_print("loading base model")
    model = TextMonkeyLMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        ignore_mismatched_sizes=True
    )

    tokenizer = QWenTokenizer.from_pretrained(
        "monkey_model",
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.IMG_TOKEN_SPAN = training_args.n_queries

    if  training_args.resampler:
        tokenizer.IMG_TOKEN_SPAN =training_args.remain
    if training_args.use_global: 
        tokenizer.IMG_TOKEN_SPAN += training_args.n_queries
    tokenizer.pad_token_id = tokenizer.eod_id
    rank0_print(tokenizer.IMG_TOKEN_SPAN)
    config.visual["n_queries"]= tokenizer.IMG_TOKEN_SPAN 
    
    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
            model.transformer.visual.requires_grad_(False)
            if not training_args.fix_resampler and  hasattr(model.transformer.visual,'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
                model.transformer.visual.ln_post.requires_grad_(True)
                model.transformer.visual.proj.requires_grad_(True)

            if hasattr(model.transformer.visual,'downresampler'):
                model.transformer.visual.downresampler.requires_grad_(True)
            for k,v in model.named_parameters():
                if "lora" in k :
                    v.requires_grad_(True)
            for k,v in model.named_parameters():
                if "window_attention" in k :
                    v.requires_grad_(True)
        if training_args.fix_llm and hasattr(model,'transformer') and hasattr(model.transformer,'h'):
            model.transformer.h.requires_grad_(False)
            model.transformer.wte.requires_grad_(False)
            model.transformer.ln_f.requires_grad_(False)
            model.lm_head.requires_grad_(False)

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = []
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    
    
    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    print_trainable_params(model)
    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)

import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
if __name__ == "__main__":
    setup_seed(46)
    train()
