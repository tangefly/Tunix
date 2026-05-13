import torch

IGNORE_INDEX = -100


class SFTCollator:

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):

        max_len = max(
            len(x["input_ids"])
            for x in features
        )

        input_ids = []
        labels = []
        attention_mask = []
        mm_token_type_ids = []

        for x in features:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(
                x["input_ids"]
                + [self.pad_token_id] * pad_len
            )

            labels.append(
                x["labels"]
                + [IGNORE_INDEX] * pad_len
            )

            attention_mask.append(
                x["attention_mask"]
                + [0] * pad_len
            )

            if "mm_token_type_ids" in x:
                mm_token_type_ids.append(
                    x["mm_token_type_ids"]
                    + [0] * pad_len
                )

        batch = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

        if mm_token_type_ids:
            batch["mm_token_type_ids"] = torch.tensor(mm_token_type_ids)

        # Handle multimodal features: pixel_values and image_grid_thw
        if any("pixel_values" in f for f in features):
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in features if "pixel_values" in f],
                dim=0,
            )

        if any("image_grid_thw" in f for f in features):
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"] for f in features if "image_grid_thw" in f],
                dim=0,
            )

        return batch
