from huggingface_hub import snapshot_download
import json

model_dir = '../models/llama-7b'
base_model = "decapoda-research/llama-7b-hf" #@param {type:"string"}
snapshot_download(repo_id=base_model, local_dir=model_dir, ignore_patterns=["LICENSE", "README.md", ".gitattributes"])

patch_model_config = True #@param {type:"boolean"}

if patch_model_config:
    replacements = {
        "tokenizer_config.json": {
            "tokenizer_class": "LlamaTokenizer",
            "model_max_length": 2048,
            "padding_side": "left",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "clean_up_tokenization_spaces": False,
            "special_tokens_map_file": "special_tokens_map.json",
        },
        "special_tokens_map.json": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "sep_token": "<s>",
            "unk_token": "<unk>",
        },
        "generation_config.json": {
            "_from_model_config": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2
        },
    }
  
    print('Patching model config...')
    for filename, new_content in replacements.items():
        print(f'{filename}:')
        with open (f'{model_dir}/{filename}') as fp:
            old_content = json.load(fp)
            print(f'    Original content: {old_content}')
            if old_content == new_content:
                print('    Already patched, skipping')
        print(f'    Updated content:  {new_content}')
        with open (f'{model_dir}/{filename}','w') as fp:
            json.dump(new_content, fp, indent=4)


#@title Уменьшаем размер батча и лимит токенов, чтобы поместиться в Colab, и длительность обучения для демки

original_config_path =  'configs/saiga_7b.json'

with open(original_config_path,'r') as fp:
    config = json.load(fp)

# Colab adjustments
config['trainer']['per_device_train_batch_size'] = 2 #@param {type:"integer"}
config['trainer']['gradient_accumulation_steps'] = 64 #@param {type:"integer"}
config['max_tokens_count'] = 1000 #@param {type:"integer"}
config['model_name'] = str(model_dir)

# Demo adjustments
config['trainer']['eval_steps'] = 2 #@param {type:"integer"}
config['trainer']['logging_steps'] = 1 #@param {type:"integer"}
config['trainer']['num_train_epochs'] = 1 #@param {type:"integer"}

config_path = 'configs/saiga_7b_colab.json'

with open(config_path,'w') as fp:
    json.dump(config, fp, indent=4)