import torch
from torch.utils.data import Dataset
import numpy as np

class GPTDataset(Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        dix = torch.tensor(chunk.astype(np.int64), dtype=torch.long)
        x = dix[:-1]
        y = dix[1:]
        return x,y