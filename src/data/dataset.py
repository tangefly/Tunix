import json
import os
from torch.utils.data import Dataset

class SFTDataset(Dataset):

    def __init__(
        self,
        data,
        processor,
    ):

        if data and isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                raise FileNotFoundError(data)
        else:
            self.data = data
        self.processor = processor

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        example = self.data[idx]

        return self.processor.process(example)