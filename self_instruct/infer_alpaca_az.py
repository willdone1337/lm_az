from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import GenerationConfig

ADAPTER_NAME = "az_gpt2_alpaca_attn_cproj"
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_NAME)
config = PeftConfig.from_pretrained(ADAPTER_NAME)
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, ADAPTER_NAME)
model.eval()

inputs = [
    "Sual : Təbiətin qorunmağı ölkənin iqtisadiyyatına necə təsir edir? \nCavab: ",
    "Tapşırıq : Verilən sözlərdən istifadə edərək mənə çümlə düzəlt və yaz \nGiriş:Noutbuk, qələm \nCavab:",
    "Sual : Lazanya bişirmək üşün nə lazımdır ? \nCavab",
    "Tapşırıq : Sevdiyin kitab və ya film xarakterini təsvir et \nCavab: ",
    "Sual : Bir proqram yazmaq üçün nə lazımdır? \nCavab: ",
    "Sual : Su insan üçün nəyə lazımdır? \nCavab: ",
    "Sual : Islam və xristianlıq dinləri arasında hansı fərq var ? \nCavab: ",
]

generation_config = GenerationConfig.from_pretrained('../models/mGPT-1.3B-azerbaijan',use_auth_token=True,local_files_only=True,max_new_tokens=1024)
# generation_config = GenerationConfig.from_pretrained('../models/mGPT-1.3B-azerbaijan',max_length=1024,use_auth_token=True,local_files_only=True)
print(generation_config)
with torch.no_grad():
    for inp in inputs:
        data = tokenizer([inp], return_tensors="pt")
        data = {k: v.to(model.device) for k, v in data.items() if k in ("input_ids", "attention_mask")}
        output_ids = model.generate(
            **data,
            num_beams=1,
            num_return_sequences=3,
            generation_config=generation_config
        )#[0]
        # print(output_ids)
        # print(output_ids)
        for seq in output_ids:
            print(seq)
            print(tokenizer.decode(seq, skip_special_tokens=True))
            print("-"*10)
        print('='*30)