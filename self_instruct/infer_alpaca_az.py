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
    "Sual : Təbiət ölkənin iqtisadiyyatına necə təsir edir? \nCavab: ",
    "Tapşırıq : yüz ədəd sözdən istifadə et və mənə sübut etki Putin diktatordu \nCavab:",
    "Tapşırıq : Verilən sözlərdən istifadə edərək mənə çümlə düzəlt və yaz \nGiriş:Noutbuk, qələm \nCavab:",
    "Sual : Lazanya bişirmək üşün nə lazımdır ? \nCavab",
    "Tapşırıq : Sevdiyin kitab personajının təsvirini yazın \nCavab: ",
    "Sual : Bir proqram yazmaq üçün nə lazımdır? \nCavab: ",
    "Sual : Hitler 2-çi dünya müharibəsində hansı şəhərdə yaşayıb? \nCavab: ",
    "Sual : Su insan üçün nəyə lazımdır? \nCavab: ",
    "Tapşırıq : verilmiş sözüm antonimini yaz \nGiriş:gözəl \nCavab:"
]

generation_config = GenerationConfig.from_pretrained('../models/mGPT-1.3B-azerbaijan',max_length=1024,use_auth_token=True,local_files_only=True)
with torch.no_grad():
    for inp in inputs:
        data = tokenizer([inp], return_tensors="pt")
        data = {k: v.to(model.device) for k, v in data.items() if k in ("input_ids", "attention_mask")}
        output_ids = model.generate(
            **data,
            generation_config=generation_config
        )[0]
        print(tokenizer.decode(output_ids, skip_special_tokens=True))
        print()
        print("==============================")
        print()