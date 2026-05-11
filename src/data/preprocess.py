from transformers import AutoTokenizer

IGNORE_INDEX = -100


class SFTProcessor:

    def __init__(
        self,
        model_name,
        max_length=4096,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.max_length = max_length

    def process(self, example):

        input_ids = []
        labels = []

        messages = example["messages"]

        # 逐轮处理
        for message in messages:

            role = message["role"]

            if role == "user":
                text = self.tokenizer.apply_chat_template(
                    [message],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                token_ids = self.tokenizer(
                    text,
                    add_special_tokens=False,
                ).input_ids

                labels.extend(
                    [IGNORE_INDEX] * len(token_ids)
                )

            elif role == "assistant":

                text = message["content"] + f"{self.tokenizer.eos_token}\n"

                token_ids = self.tokenizer(
                    text,
                    add_special_tokens=False,
                ).input_ids

                labels.extend(token_ids)

            else:
                raise ValueError(f"Unknown role: {role}")

            input_ids.extend(token_ids)

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }