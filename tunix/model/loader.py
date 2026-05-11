import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)

def load_model(model_path):
    
    model_config = AutoConfig.from_pretrained(model_path)

    processor = None
    if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
        load_class = AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_path)
    elif type(model_config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
        load_class = AutoModelForSeq2SeqLM
    elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text for qwen omni
        load_class = AutoModelForTextToWaveform
    else:
        load_class = AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.bfloat16
    else:
        device_map = None
        dtype = torch.float32

    model = load_class.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, processor