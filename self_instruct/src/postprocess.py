import argparse
import random
import json
import os

import wandb
import torch
import numpy as np
import bitsandbytes as bnb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, logging, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from src.dataset import InstructDataset, ChatDataset
from src.util.dl import set_random_seed, fix_tokenizer, fix_model
from src.util.io import read_jsonl

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
