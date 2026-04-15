import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset

# Import your custom modules
from config import GPTConfig
from model import GPT

# ==========================================
# 1. Dataset with Strict Response Masking
# ==========================================
class LIMADataset(Dataset):
    def __init__(self, block_size=1024):
        # Using Alpaca-Cleaned to avoid the gated dataset error
        self.dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.IGNORE_INDEX = -100 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        user_text = item['instruction'] + (f"\n{item['input']}" if item['input'] else "")
        assistant_text = item['output']
        
        prompt_str = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        response_str = f"{assistant_text}<|im_end|>"
        
        prompt_tokens = self.enc.encode(prompt_str, allowed_special="all")
        response_tokens = self.enc.encode(response_str, allowed_special="all")
        
        full_tokens = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        
        if len(full_tokens) > self.block_size:
            full_tokens = full_tokens[:self.block_size]
            
        x = torch.tensor(full_tokens[:-1], dtype=torch.long)
        y = torch.tensor(full_tokens[1:], dtype=torch.long)
        
        y[:prompt_len - 1] = self.IGNORE_INDEX
        
        actual_len = len(x)
        if actual_len < (self.block_size - 1):
            padding_len = (self.block_size - 1) - actual_len
            x = torch.cat([x, torch.full((padding_len,), self.enc.eot_token, dtype=torch.long)])
            y = torch.cat([y, torch.full((padding_len,), self.IGNORE_INDEX, dtype=torch.long)])
            
        return x, y

# ==========================================
# 2. Main Training Loop (Single GPU)
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = GPTConfig()
    model = GPT(config)
    
    checkpoint_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state_dict)
        print(f"✅ Loaded base model.")

    model.to(device)
    
    dataset = LIMADataset(block_size=config.block_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=True) 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
    
    model.train()
    print(f"🚀 Starting SFT on {device}...")

    for epoch in range(3):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "/kaggle/working/gpt250m_sft_final.pth")
    print("🎉 Done!")

if __name__ == "__main__":
    main()