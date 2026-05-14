from torch.utils.data import DataLoader

from tunix.data.preprocess import SFTProcessor
from tunix.data.dataset import SFTDataset
from tunix.data.collator import SFTCollator


data = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "/home/tanger/workspace/Tunix/data/robot/images/ameca_9.png",
                    },
                    {
                        "type": "image",
                        "image": "/home/tanger/workspace/Tunix/data/robot/images/leju_kuafu_2.png",
                    },
                    {
                        "type": "text",
                        "text": "分析这两张图片！",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "这是一个很好的问题！我们可以从几个方面来思考这个问题。"
            }
        ]
    }
]

processor = SFTProcessor(
    model_name="/home/tanger/workspace/models/Qwen3-VL-4B-Instruct",
)

dataset = SFTDataset(
    data=data,
    processor=processor,
)

collator = SFTCollator(
    processor.tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=collator,
)

batch = next(iter(loader))

print(batch)

print()

print(
    processor.tokenizer.decode(
        batch["input_ids"][0]
    )
)