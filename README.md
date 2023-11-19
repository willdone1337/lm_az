# Alpaca-Az

<p align="center">
  <img src="repo_image/llama_aerodrom.png" alt="Your Image" width="350" style="border-radius:50%;">
</p>

## Introduction
This repository builds upon the [ru_lrm](https://github.com/IlyaGusev/rulm) repository and the [mGPT-Azerbaijan](https://huggingface.co/ai-forever/mGPT-1.3B-azerbaijan) model with specific adaptations for training and applying a model in Azerbaijani language. Trained adapter [weights](https://huggingface.co/vildanh/az_gpt_alpaca) and [dataset](https://huggingface.co/datasets/vildanh/az_alpaca_translated) are here. Before diving in, it's crucial to provide some context about the current state of the model.


This language model is a work in progress, and wanted to be clear about its current limitations. At this stage:

- **Accuracy:** The model's accuracy may not be optimal, and it might produce outputs that contain errors or inaccuracies.
  
- **Hallucinations:** The model may exhibit hallucination-like behavior, generating content that is not contextually accurate or coherent.

Acknowledging these limitations is part of the ongoing process of refining and evolving the model. The intent is to create an open space for collaboration, where the community can actively contribute to enhancing the model's performance. In this experiment, also utilized the Llama model for fine-tuning. However, it's remarkable that the mGPT model trained for Azerbijani language demonstrated higher accuracy during training, and shared weights belonged to it.

## How You Can Contribute

Your contributions and feedback are crucial for refining and evolving this language model. Here's how you can contribute:

1. **Data Quality:** Provide high-quality Azerbaijani language data to improve the model's training dataset.
  
2. **Model Tuning:** Contribute to fine-tuning strategies, model architecture enhancements, or propose new approaches to mitigate hallucinations.

3. **Feedback:** Share your experiences, report issues, and provide constructive feedback on the model's outputs.



## Steps Taken

1. **Translation of Dataset:**
   - Utilized the Google Translate API to translate the `ru_turbo_alpaca` dataset to Azerbaijani.
   - Resulted in a dataset comprising ~30k question and answer pairs.

2. **Model Replacement:**
   - Substituted the llama model with the `mgpt-az` model, featuring 1.3 billion parameters.
   - Modified the input template of the dataset for compatibility.
   - Change the trainable adapters from **k/v/q/_proj** to **c_proj, c_attn**

### Example Input Template Modification

```plaintext
{
    "description": "Template used by Azerbaijan Alpaca-LoRA.",
    "prompts_input": [
        "### Tapşırıq: {instruction}\n### Giriş: {inp}\n### Cavab: "
    ],
    "prompts_no_input": [
        "### Tapşırıq: {instruction}\n### Cavab: "
    ],
    "output_separator": "Cavab: "
}
```

## Training Details

- For Llama finetuning, components such as k_proj, v_proj, etc., were utilized. In this context, c_proj or attn_proj was employed.
- Training took place on a GPU for a few hours, utilizing a batch size of 4 and over 3 epochs.
- The training loss reached 1.02, while the evaluation loss is 0.96.

## Examples
```
Sual : Təbiətin qorunmağı ölkənin iqtisadiyyatına necə təsir edir? 
Cavab: Təbiəti qorumağın bir çox yolu var, o cümlədən ekoloji cəhətdən təmiz texnika və müntəzəm nəqliyyat vasitələri ilə. Bununla belə, hökumət tərəfindən təşviq edilə biləcək daha az miqyaslı ekoloji problemlər də var. Məsələn, tullantıların toplanması və utilizasiyası üçün bəzi yerlərdə elektrik enerjisi kimi alternativ enerji mənbələrinə keçidlə bağlı problemlər də ola bilər. Bundan əlavə, təbiətin mühafizəsi sahəsində fəaliyyət göstərən beynəlxalq təşkilatlar da bu problemin öhdəsindən gəlməlidirlər.
```
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
    ## Installation
    ```bash
    git clone https://github.com/willdone1337/lm_az
    cd rulm
    pip install -r requirements.txt
    ```
 - Download and move the mGPT-1.3B-azerbaijan model to the /lm_az/models/ directory
    ```bash
    cd lm_az
    python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="ai-forever/mGPT-1.3B-azerbaijan", local_dir="models/mGPT-1.3B-azerbaijan")'
    ```

 - Download and move the az_gpt2_alpaca to /lm_az/self_instruct directory
    ```bash
    cd self_instruct
    python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="vildanh/az_gpt_alpaca", local_dir="az_gpt2_alpaca_attn_cproj/")'
    ```
 - alpaca_az_translated is exist in the src/data_processing dir

### Inference
```bash
python3 infer_alpaca_az.py 
```
### Fine-Tune
```bash
cd self_instruct &&  python3 -m src.train --config-file configs/llama_7b_lora.json --train-file src/data_processing/alpaca_az_read_edited_v2.jsonl --val-file src/data_processing/alpaca_az_read_eval_edited_v2.jsonl --output-dir az_gpt2_alpaca
```

