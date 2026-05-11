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

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }