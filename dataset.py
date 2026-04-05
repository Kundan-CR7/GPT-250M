import torch
import json
from torch.utils.data import Dataset,DataLoader
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

class DPODataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the preference dataset
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the text (e.g., adding prompt + response)
        chosen_text = f"Prompt: {item['prompt']}\nAnswer: {item['chosen']}<|endoftext|>"
        rejected_text = f"Prompt: {item['prompt']}\nAnswer: {item['rejected']}<|endoftext|>"
        
        # Tokenize chosen
        chosen_enc = self.tokenizer(
            chosen_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # Tokenize rejected
        rejected_enc = self.tokenizer(
            rejected_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # Squeeze to remove the batch dimension added by the tokenizer
        return {
            "chosen_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_mask": rejected_enc["attention_mask"].squeeze(0)
        }

def get_dpo_dataloader(jsonl_path, tokenizer, batch_size=4, max_length=512):
    dataset = DPODataset(jsonl_path, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)