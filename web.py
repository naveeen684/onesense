import gradio as gr
from transformers import pipeline
import numpy as np
import os
import warnings

from modules.logging_colors import logger
from modules.block_requests import OpenMonkeyPatch, RequestBlocker

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

with RequestBlocker():
    import gradio as gr

import matplotlib
matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import importlib
import json
import math
import os
import re
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from threading import Lock

import psutil
import torch
import yaml
from PIL import Image

import modules.extensions as extensions_module
from modules import chat, loaders, presets, shared, training, ui, utils
from modules.extensions import apply_extensions
from modules.github import clone_or_pull_repository
from modules.html_generator import chat_html_wrapper
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_settings_from_yamls,
    save_model_settings,
    update_model_parameters
)
from modules.text_generation import (
    generate_reply_wrapper,
    get_encoded_length,
    stop_everything_event
)
from modules.utils import gradio
from extensions.whisper_stt.script import do_stt
import speech_recognition as sr
import whisper
    

def transcribe(audio):
    sr, y = audio
    print(audio)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    print(y)
    model = whisper.load_model("small")


    processed_text = model.transcribe(y, task = "translate")
    print(processed_text)
    processed_text = "FInd whether the given sentence is a prescription or not and then extract and create a prescription table with medicine names (correct names if it is incorrect), dosage and time from the below sentence only if it contains prescription else say no prescription found"+processed_text["text"]    
    print(processed_text)
    state = {"max_new_tokens": 200, "seed": -1.0, "temperature": 0.7, "top_p": 0.9, "top_k": 20, "typical_p": 1, "epsilon_cutoff": 0, "eta_cutoff": 0, "repetition_penalty": 1.15, "repetition_penalty_range": 0, "encoder_repetition_penalty": 1, "no_repeat_ngram_size": 0, "min_length": 0, "do_sample": True, "penalty_alpha": 0, "num_beams": 1, "length_penalty": 1, "early_stopping": False, "mirostat_mode": 0, "mirostat_tau": 5, "mirostat_eta": 0.1, "add_bos_token": True, "ban_eos_token": False, "truncation_length": 4096, "custom_stopping_strings": "", "skip_special_tokens": True, "stream": True, "tfs": 1, "top_a": 0, "character_menu": "None", "history": {"internal": [["Hello", "Hello! I'm here to help answer your questions. What would you like to know?"]], "visible": [["Hello", "Hello! I'm here to help answer your questions. What would you like to know?"]]}, "name1": "You", "name2": "Assistant", "greeting": "", "context": "This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.", "chat_generation_attempts": 1, "stop_at_newline": False, "mode": "instruct", "instruction_template": "Llama-v2", "name1_instruct": "", "name2_instruct": "", "context_instruct": "[INST] <<SYS>>\nAnswer the questions.\n<</SYS>>\n", "turn_template": "<|user|><|user-message|> [/INST] <|bot|><|bot-message|> </s><s>[INST] ", "chat_style": "cai-chat", "chat-instruct_command": "Continue the chat dialogue below. Write a single reply for the character \"<|character|>\".\n\n<|prompt|>", "loader": "Transformers", "cpu_memory": 0, "auto_devices": False, "disk": False, "cpu": False, "bf16": False, "load_in_8bit": True, "trust_remote_code": False, "load_in_4bit": False, "compute_dtype": "float16", "quant_type": "nf4", "use_double_quant": False, "wbits": "None", "groupsize": "None", "model_type": "llama", "pre_layer": 0, "triton": False, "desc_act": False, "no_inject_fused_attention": False, "no_inject_fused_mlp": False, "no_use_cuda_fp16": False, "threads": 0, "n_batch": 512, "no_mmap": False, "low_vram": False, "mlock": False, "n_gpu_layers": 0, "n_ctx": 2048, "llama_cpp_seed": 0.0, "gpu_split": "", "max_seq_len": 2048, "compress_pos_emb": 1, "alpha_value": 1, "gpu_memory_0": 0} 
    generators = chat.generate_chat_reply(processed_text,  state)           
    for generator in generators:
        processed_text=generator['visible'][-1][-1]
        print("Result", processed_text)
        yield processed_text

if __name__ == "__main__":
    # Loading custom settings
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif Path('settings.yaml').exists():
        settings_file = Path('settings.yaml')
    elif Path('settings.json').exists():
        settings_file = Path('settings.json')

    if settings_file is not None:
        logger.info(f"Loading settings from {settings_file}...")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        for item in new_settings:
            shared.settings[item] = new_settings[item]

    # Set default model settings based on settings file
    shared.model_config['.*'] = {
        'wbits': 'None',
        'model_type': 'None',
        'groupsize': 'None',
        'pre_layer': 0,
        'mode': shared.settings['mode'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
        'truncation_length': shared.settings['truncation_length'],
    }

    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Default extensions
    extensions_module.available_extensions = utils.get_available_extensions()
    if shared.is_chat():
        for extension in shared.settings['chat_default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)
    else:
        for extension in shared.settings['default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':
        model_settings = get_model_settings_from_yamls(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults
        update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    # Forcing some events to be triggered on page load
    shared.persistent_interface_state.update({
        'loader': shared.args.loader or 'Transformers',
    })

    if shared.is_chat():
        shared.persistent_interface_state.update({
            'mode': shared.settings['mode'],
            'character_menu': shared.args.character or shared.settings['character'],
            'instruction_template': shared.settings['instruction_template']
        })

        if Path("cache/pfp_character.png").exists():
            Path("cache/pfp_character.png").unlink()

    shared.generation_lock = Lock()

    demo = gr.Interface(
      transcribe,
      gr.Audio(sources=["microphone"]),
      "text",
      live=True
    )

    demo.queue(concurrency_count=5, max_size=20).launch(share=True, debug=True)

