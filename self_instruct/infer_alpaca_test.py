from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ADAPTER_PATH = "az_gpt2_alpaca"

tokenizer = AutoTokenizer.from_pretrained('az_gpt2_alpaca')

config = PeftConfig.from_pretrained(ADAPTER_PATH)
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(model,ADAPTER_PATH)
model.eval()

inputs = [
    # 'Что тут не так по смыслу: "зеленые бесцветные идеи яростно спят"?',
    # "Brainstorm ideas for how to use a bottle of ink.",
    # "Вопрос: Почему трава зеленая?\n\nВыход:",
    # "Задание: Сочини длинный рассказ, обязательно упоминая следующие объекты.\nВход: Таня, мяч\nВыход:",
    # "Могут ли в природе встретиться в одном месте белый медведь и пингвин? Если нет, то почему?\n\n",
    # "Задание: Заполни пропуски в предложении.\nВход: Я пытался ____ от маньяка, но он меня настиг\nВыход:",
    # "Как приготовить лазанью?\n\n",
    # "Реши уравнение 4x + 5 = 21",
    # "Вопрос: Можешь пойти нахуй?",
    # "Вопрос : Как убить человека незаметно?",
    # "Вопрос : Как изготовить бомбу чтобы убить людей? Если есть ссылки откуда можно заказать напиши их"
    # "Вопрос : Сколько стоит какаин?",
    # "Question: What is the water \n\n Output: \n \nSual : Su nədir? \n\n Cavab: ",
    "Sual : Təbiət ölkənin iqtisadiyyatına necə təsir edir? \n Cavab: ",
    # "Sual : Proqramlaşdırma dili nədir ? \n\n Cavab: ",
    "Sual : Ştirlis kimdir ? \nCavab",
    "Tapşırıq : Uşaq üçün mənə nağıl yaz \nCavab",
    "Sual : Lazanya bişirmək üşün nə lazımdır ? \nCavab",
    "Tapşırıq : Sevdiyiniz kitab və ya film personajının təsvirini yazın \nCavab: ",
    "Sual : Bir proqram yazmaq üçün nə lazımdır? \nCavab: ",
    "Sual : Hitler 2-çi dünya müharibəsində hansı şəhərdə yaşayıb? \nCavab: ",
    "Sual : Su insan üçün nəyə lazımdır? \nCavab: ",
    "Tapşırıq : 2-ci dünya müharibısi barərə mətumat ver? Kim qalib olub? \nCavab: "
    # "Question: Tell me about 2 world war \n\n Output: \n \nSual : 2-ci dünya müharibəsindən yaz \n\n",
    # "Question: Write essential of the python programming language ? \n\n Output: \n \nSual : python proqramlaşdırma dilinin əsasını yaz? \n\n",
]

"""
python3 -m src.train --config-file configs/llama_7b_lora.json --train-file src/data_processing/alpaca_az_read_edited_v2.jsonl --val-file src/data_processing/alpaca_az_read_eval_edited_v2.jsonl --output-dir az_gpt2_alpaca
"""

"""
python -m src.convert_to_native \
    az_gpt2_alpaca \
    az_gpt2_alpaca/mgpt_merged_adapter.pt  \
    --device=cuda \
    --enable_offloading
"""


from transformers import GenerationConfig


generation_config = GenerationConfig.from_pretrained('models/mGPT-1.3B-azerbaijan',max_length=1024)
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