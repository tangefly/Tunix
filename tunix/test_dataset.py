from torch.utils.data import DataLoader

from .data.preprocess import SFTProcessor
from .data.dataset import SFTDataset
from .data.collator import SFTCollator


data = [
    {
        "messages": [
            {
                "role": "user",
                "content": "为什么文艺复兴被认为是艺术史上的重要转折点？"
            },
            {
                "role": "assistant",
                "content": "这是一个很好的问题！我们可以从几个方面来思考这个问题。"
            },
            {
                "role": "user",
                "content": "不是很清楚。"
            },
            {
                "role": "assistant",
                "content": "没关系，让我们一步一步来。你能想象在中世纪，艺术作品主要描绘些什么吗？"
            }
        ]
    }
]

processor = SFTProcessor(
    model_name="/home/tanger/workspace/models/Qwen3-4B",
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