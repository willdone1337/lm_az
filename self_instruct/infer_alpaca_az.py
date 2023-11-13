from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import GenerationConfig

MODEL_NAME = "az_gpt2_alpaca"
tokenizer = AutoTokenizer.from_pretrained('az_gpt2_alpaca')
config = PeftConfig.from_pretrained(MODEL_NAME)
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=False,
    device_map="cpu"
)

model = PeftModel.from_pretrained(model, MODEL_NAME)
model.eval()

inputs = [
    "Sual : Təbiət ölkənin iqtisadiyyatına necə təsir edir? \n Cavab: ",
    "Sual : Ştirlis kimdir ? \nCavab",
    "Tapşırıq : Uşaq üçün mənə nağıl yaz \nCavab",
    "Sual : Lazanya bişirmək üşün nə lazımdır ? \nCavab",
    "Tapşırıq : Sevdiyiniz kitab və ya film personajının təsvirini yazın \nCavab: ",
    "Sual : Bir proqram yazmaq üçün nə lazımdır? \nCavab: ",
    "Sual : Hitler 2-çi dünya müharibəsində hansı şəhərdə yaşayıb? \nCavab: ",
    "Sual : Su insan üçün nəyə lazımdır? \nCavab: ",
    "Tapşırıq : 2-ci dünya müharibısi barərə mətumat ver? Kim qalib olub? \nCavab: "
]



generation_config = GenerationConfig.from_pretrained('models/llama-7b',max_length=1024)
print('~'*20)
print(generation_config)
print('~'*20)
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