import torch
from transformers import pipeline, set_seed
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained("../models/mGPT-1.3B-azerbaijan")
model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained("../models/mGPT-1.3B-azerbaijan")
text = """Qışda qar ilə"""
input_ids = tokenizer.encode(text, return_tensors="pt").cuda(device)
model.eval()
for x,y in model.named_parameters():
    print(x)


with torch.no_grad():
    out = model.generate(
            input_ids, 
            min_length=100, 
            max_length=100, 
            top_k=10,
            top_p=0.0,
            no_repeat_ngram_size=5
    )
    generated_text = list(map(tokenizer.decode, out))[0]
    print(generated_text)






















# txt = """Qışda qar ilə"""

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# pipe = pipeline(
#     "text-generation", model="../models/mGPT-1.3B-azerbaijan", device=device
#     # "text-generation", model="aze_exp_3/checkpoint-1835008", device=device
# )

# set_seed(43)
# print(pipe(txt, max_length=256, temperature=0.5,pad_token_id=pipe.tokenizer.eos_token_id))# [0]["generated_text"])
# import jsonlines

# def replace_non_utf_symbols_with_utf(text):
#     def replace_non_utf_chars(match):
#         return chr(int(match.group(1), 16))

#     import re
#     pattern = r'\\u([0-9a-fA-F]{4})'
#     return re.sub(pattern, replace_non_utf_chars, text)

# def process_jsonl_file(input_file, output_file):
#     updated_data = []

#     with jsonlines.open(input_file, 'r') as reader:
#         for line in reader:
#             if "instruction" in line:
#                 line["instruction"] = replace_non_utf_symbols_with_utf(line["instruction"])
#             updated_data.append(line)

#     with jsonlines.open(output_file, 'w') as writer:
#         writer.write_all(updated_data)

# # Example usage

# input_file = "src/data_processing/alpaca_az_read_edited.jsonl"  # Replace with your input JSONL file
# output_file = "src/data_processing/alpaca_az_read_edited_v2.jsonl"  # Replace with your output JSONL file

# process_jsonl_file(input_file, output_file)# Example usage
