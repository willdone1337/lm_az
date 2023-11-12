import argparse
import random
from itertools import chain

from datasets import load_dataset, Value, Features, Sequence
from transformers import AutoTokenizer
from tqdm import tqdm


MAX_TOKENS = 10000000


def tokenize(examples, tokenizer, position_ids):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_TOKENS,
        padding=False,
        return_length=True
    )
    outputs.pop("attention_mask")
    outputs.pop("token_type_ids")
    lengths = outputs.pop("length")
    outputs["position_ids"] = [position_ids[:l] for l in lengths]
    return outputs


def group(examples, block_size):
    concatenated_examples = {k: list(chain(*v)) for k, v in examples.items()}
    some_key = list(examples.keys())[0]
    total_length = len(concatenated_examples[some_key])

    # Remove reminder to skip padding handling
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def run(examples, tokenizer, block_size, position_ids):
    examples = tokenize(examples, tokenizer, position_ids)
    return group(examples, block_size)


def preprocess(
    dataset_path,
    tokenizer_path,
    block_size,
    streaming,
    output_path
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    datasets = load_dataset(dataset_path, streaming=streaming)

    position_ids = [i % block_size for i in range(MAX_TOKENS)]

    datasets = datasets.map(
        lambda x: run(x, tokenizer, block_size, position_ids),
        batched=True,
        remove_columns=["text"]
    ).cast(Features({
        "input_ids": Sequence(Value("uint16")),
        "position_ids": Sequence(Value("uint16"))
    }))

    datasets.save_to_disk(output_path, max_shard_size="1GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--block-size", default=512, type=int)
    parser.add_argument("--streaming", action="store_true", default=False)
    args = parser.parse_args()
    preprocess(**vars(args))
