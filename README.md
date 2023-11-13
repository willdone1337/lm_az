# Alpaca-Az

<!-- 
<div style="text-align:center">
  <img src="llama_aerodrom.png" alt="Project Image" style="border-radius:50%; width:200px; height:200px;">
</div> -->

<p align="center">
  <img src="llama_aerodrom.png" alt="Your Image" width="200" style="border-radius:50%;">
</p>
This repository builds upon the foundation of the [ru_lrm](https://github.com/IlyaGusev/rulm) repository and the [mGPT-Azerbaijan](https://huggingface.co/ai-forever/mGPT-1.3B-azerbaijan) model with specific adaptations tailored for training and applying a model in Azerbaijani language. The key steps include:

## Steps Taken

1. **Translation of Dataset:**
   - Utilized the Google Translate API to translate the `ru_turbo_alpaca` dataset to Azerbaijani.
   - Resulted in a dataset comprising 32,000 question and answer pairs.

2. **Model Replacement:**
   - Substituted the llama model with the `mgpt-az` model, featuring 1.3 billion parameters.
   - Modified the input template of the dataset for compatibility.

### Example Input Template Modification

```plaintext
{
    "description": "Template used by Azerbaijan Alpaca-LoRA.",
    "prompts_input": [
        "### Task: {instruction}\n### Input: {inp}\n### Answer: "
    ],
    "prompts_no_input": [
        "### Task: {instruction}\n### Answer: "
    ],
    "output_separator": "Answer: "
}
```
## Training Details

- For Llama finetuning, components such as k_proj, v_proj, etc., were utilized. In this context, c_proj or attn_proj was employed.
- Training took place on a GPU for a few hours, utilizing a batch size of 4 and spanning over 3 epochs.
- The training loss reached 1.03, while the evaluation loss is reported as 0.97.

## Examples
```
Sual : Su insan üçün nəyə lazımdır? 
Cavab: Su insan üçün çox vacibdir, çünki o, sağlamlığınızı və rifahınızı yaxşılaşdırmağa kömək edir. O, həmçinin ürək-damar xəstəlikləri, diabet və xərçəng kimi müxtəlif xəstəliklərin riskini azaltmağa kömək edir.
```
```
Tapşırıq : Sevdiyiniz kitab və ya film personajının təsvirini yazın 
Cavab: Ən çox sevdiyim kitab Mixail Bulqakovun “Ustad və Marqarita” kitabıdır. Bu, Moskvada Ustad və onun sevimli Marqaritanın hekayəsindən bəhs edir. Kitabın süjeti, personajları və onun mənəviyyatı məni valeh edir.
```



# Setup


 - Used nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04 image 
 with torch==2.0.0cu+117
 - Install requirements
```bash
git clone https://github.com/willdone1337/alpaca_az
cd rulm
pip install -r requirements.txt

```
 - Download and move the mGPT-1.3B-azerbaijan model to the /rulm/models/ directory
 ```bash
 python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="ai-forever/mGPT-1.3B-azerbaijan", local_dir="models/")'
 ```

 - Download and move the az_gpt2_alpaca to rulm/self_instruct directory
```bash
 python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="ai-forever/willdone1337/model", local_dir="./")'
 ```
 - alpaca_az_translated is exist in the src/data_processing dir
 - Provide the model in the inference py script

### Inference
```bash
python3 infer_alpaca_az.py 
```
### Fine-Tune
```bash
cd self_instruct &&  python3 -m src.train --config-file configs/llama_7b_lora.json --train-file src/data_processing/alpaca_az_read_edited_v2.jsonl --val-file src/data_processing/alpaca_az_read_eval_edited_v2.jsonl --output-dir az_gpt2_alpaca
```