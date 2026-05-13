from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)

IGNORE_INDEX = -100

class SFTProcessor:

    def __init__(
        self,
        model_name,
        max_length=4096,
    ):
        
        self.processor = self.get_processor(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.max_length = max_length

    def get_processor(self, model_name):
        """
            get processor for mLLM
        """
        model_config = AutoConfig.from_pretrained(model_name)

        processor = None
        if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
            processor = AutoProcessor.from_pretrained(model_name)
        elif type(model_config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
            processor = AutoProcessor.from_pretrained(model_name)
        elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text
            processor = AutoProcessor.from_pretrained(model_name)
        
        return processor

    def process(self, example):

        input_ids = []
        labels = []

        messages = example["messages"]

        if not self.processor:
            for message in messages:
                role = message["role"]
                if role == "user":
                    text = self.tokenizer.apply_chat_template(
                        [message],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                    labels.extend([IGNORE_INDEX] * len(token_ids))
                elif role == "assistant":
                    text = message["content"] + f"{self.tokenizer.eos_token}\n"
                    token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                    labels.extend(token_ids)
                else:
                    raise ValueError(f"Unknown role: {role}")

                input_ids.extend(token_ids)
        else:
            input_ids = []
            for message in messages:
                role = message["role"]
                if role == "user":
                    inputs = self.processor.apply_chat_template(
                        [message],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                    )
                    input_ids.extend(inputs["input_ids"][0])
                    labels.extend([IGNORE_INDEX] * len(inputs["input_ids"]))
                elif role == "assistant":
                    text = message["content"] + f"{self.tokenizer.eos_token}\n"
                    token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                    input_ids.extend(token_ids)
                    labels.extend(token_ids)
                else:
                    raise ValueError(f"Unknown role: {role}")
                
                inputs["input_ids"] = input_ids

        # input_ids = input_ids[:self.max_length]
        # labels = labels[:self.max_length]
        attention_mask = [1] * len(inputs["input_ids"])

        return {
            **inputs,
            "labels": labels,
            "attention_mask": attention_mask,
        }