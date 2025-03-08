# Orpheus TTS
## Overview

Orpheus TTS is an open-source text-to-speech system built on the Llama-3b backbone, Orpheus demonstrates the emergent capabilities of using LLMs for speech synthesis. We offer comparisons of the models below to leading closed models like Eleven Labs and PlayHT in our blog post.

## Emergent Abilities

- **Human-Like Speech**: Natural intonation, emotion, and rhythm that is superior to SOTA closed source models
- **Zero-Shot Voice Cloning**: Clone voices without prior training or fine-tuning
- **Guided Emotion and Intonation**: Control speech characteristics with simple tags
- **Low Latency**: ~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming

## Models

We provide three models in this release, for the two finetunes we offer data processing scripts and sample datasets
to make it very straightforward to create your own finetune.

1. [**Pretrained**](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) Our base model trained on 100k+ hours of English speech data
2. [**Finetuned Regular**](https://huggingface.co/canopylabs/orpheus-tts-0.1-primary) A finetuned model for everyday TTS applications
3. [**Finetuned Tags**](https://huggingface.co/canopylabs/orpheus-tts-0.1-emo-instruct) Enhanced model with emotional expression capabilities

### Inference

1. [Colab For Pretrained Model](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing) This notebook is set up for conditioned generation but can be extended to a range of tasks.
2. [Colab For Tuned Model](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing) A finetuned model for everyday TTS applications

#### Prompting

1. For the pretrained model, you can either generate speech just conditioned on text, or generate speech conditioned on one or more existing text-speech pairs in the prompt. Since this model hasn't been explicitly trained on the zero-shot voice cloning objective the more text-speech pairs you pass in the prompt, the more reliably it will generate in the correct voice.

2. For the finetuned models: For the primary model pass in <zac> or <zoe> at the end, after a space, to indicate the voice. For the model trained on tags pass in <emotion> or <artefact/>, check in emotions.txt for the various emotions and artefacts the model has seen.



#### Realtime Inference

1. Clone This Repo
   ```bash
   git clone https://github.com/canopyai/Orpheus-TTS-0.1.git
   ```
2. Navigate and install packages
   ```bash
   cd vllm_inference

   pip install flask flask-cors snac vllm
   ```
3. Start the server
   ```bash
   python main.py
   ```

4. Change the Url you connect to in vllm_inference/client.html

## Finetune Model

Here is an overview of how to finetune your model on any text and speech.
This is a very simple process analogous to tuning an LLM using Trainer and Transformers

1. Your dataset should be a huggingface dataset in [this format](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
2. First we tokenise your dataset using [this notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing). This pushes an intermediate dataset to your Hugging Face account which you will use for the second stage of processing. You will enter the name of your dataset, the namespace of the intermediate dataset, and a HF write token in the notebook. This should take less than 1 minute/thousand rows.
3. Clone this repo, and run edit the namespaces listed at the top of preprocess.py so it links to the dataset the notebook in step 2 pushed. This should take less than 1 minute/100k rows.
   ```bash
   cd finetune
   pip install datasets transformers
   huggingface-cli login <enter your write token>
   python preprocess.py
   ```
4. Modify the finetune/config.yaml file to include your dataset and training properties, and run the training script. You can additionally run any kind of huggingface compatible process like Lora to tune the model.
   ```bash
    pip install transformers datasets wandb trl flash_attn torch
    huggingface-cli login <enter your write token>
    wandb login <wandb token>
    accelerate launch train.py
   ```


