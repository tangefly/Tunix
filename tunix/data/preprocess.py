import os
import torch

from transformers import (
    AutoConfig,
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
        image_base_dir=None,
    ):
        self.processor = self.get_processor(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.image_base_dir = image_base_dir

        # Limit image resolution to control visual token count and avoid OOM.
        # Qwen3-VL uses Qwen2VLImageProcessor which controls resolution via
        # size={"longest_edge": ..., "shortest_edge": ...}, NOT max_pixels.
        # Visual tokens ≈ (longest_edge/patch_size)^2 / merge_size^2
        #   with patch_size=16, merge_size=2.
        if self.processor is not None and hasattr(self.processor, "image_processor"):
            self.processor.image_processor.size["longest_edge"] = 262144   # 512px → ~256 visual tokens
            self.processor.image_processor.size["shortest_edge"] = 65536   # 256px

    def get_processor(self, model_name):
        """get processor for mLLM"""
        model_config = AutoConfig.from_pretrained(model_name)

        processor = None
        if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():
            processor = AutoProcessor.from_pretrained(model_name)
        elif type(model_config) in AutoModelForSeq2SeqLM._model_mapping.keys():
            processor = AutoProcessor.from_pretrained(model_name)
        elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():
            processor = AutoProcessor.from_pretrained(model_name)

        return processor

    def _truncate(self, result):
        """Truncate sequence to max_length."""
        for key in ("input_ids", "labels", "attention_mask", "mm_token_type_ids"):
            if key in result and len(result[key]) > self.max_length:
                result[key] = result[key][:self.max_length]
        return result

    def _resolve_image_path(self, path):
        """Resolve relative image path to absolute path."""
        if os.path.isabs(path):
            return path
        if self.image_base_dir:
            return os.path.normpath(os.path.join(self.image_base_dir, path))
        return path

    def _convert_old_format(self, example):
        """
        Convert old data format to Qwen VL format.

        Old format:
            {"messages": [{"role": "user", "content": "<image>text..."}, ...],
             "images": ["path/to/img.png"]}

        New format:
            {"messages": [{"role": "user", "content": [
                {"type": "image", "image": "path/to/img.png"},
                {"type": "text", "text": "text..."}
            ]}, ...]}
        """
        messages = example.get("messages", [])
        images = example.get("images", [])

        if not images:
            return messages

        image_idx = 0
        new_messages = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                content = msg["content"]
                new_content = []
                parts = content.split("<image>")
                for i, part in enumerate(parts):
                    if part:
                        new_content.append({"type": "text", "text": part})
                    if i < len(parts) - 1 and image_idx < len(images):
                        img_path = self._resolve_image_path(images[image_idx])
                        new_content.append({"type": "image", "image": img_path})
                        image_idx += 1
                msg = {**msg, "content": new_content}
            new_messages.append(msg)

        return new_messages

    def process(self, example):
        messages = example["messages"]

        if not self.processor:
            # Text-only path
            input_ids = []
            labels = []
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

            attention_mask = [1] * len(input_ids)
            result = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
            return self._truncate(result)

        # Multimodal path (VL models)
        # Note: do NOT truncate here — truncating input_ids would break the
        # alignment with pixel_values / image_grid_thw. Visual tokens are
        # controlled via image_processor.size instead. (VL models)
        # Convert old format if needed
        if any(isinstance(msg["content"], str) for msg in messages if msg["role"] == "user"):
            if "images" in example:
                messages = self._convert_old_format(example)

        input_ids = []
        labels = []
        mm_token_type_ids = []
        pixel_values_list = []
        image_grid_thw_list = []

        for message in messages:
            role = message["role"]
            if role == "user":
                inputs = self.processor.apply_chat_template(
                    [message],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )
                seq_len = len(inputs["input_ids"][0])
                input_ids.extend(inputs["input_ids"][0])
                labels.extend([IGNORE_INDEX] * seq_len)

                if "mm_token_type_ids" in inputs:
                    mmt = inputs["mm_token_type_ids"][0]
                    if isinstance(mmt, list):
                        mm_token_type_ids.extend(mmt)
                    else:
                        mm_token_type_ids.extend(mmt.tolist())
                else:
                    mm_token_type_ids.extend([0] * seq_len)

                if "pixel_values" in inputs:
                    pixel_values_list.append(inputs["pixel_values"])
                if "image_grid_thw" in inputs:
                    image_grid_thw_list.append(inputs["image_grid_thw"])
            elif role == "assistant":
                text = message["content"] + f"{self.tokenizer.eos_token}\n"
                token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                input_ids.extend(token_ids)
                labels.extend(token_ids)
                mm_token_type_ids.extend([0] * len(token_ids))
            else:
                raise ValueError(f"Unknown role: {role}")

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "mm_token_type_ids": mm_token_type_ids,
        }

        # Combine pixel_values and image_grid_thw from all user messages
        if pixel_values_list:
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        attention_mask = [1] * len(input_ids)
        result["attention_mask"] = attention_mask

        return result
