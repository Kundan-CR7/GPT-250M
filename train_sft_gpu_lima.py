import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken
from datasets import load_dataset

# Import your custom modules
from config import GPTConfig
from model import GPT

# ==========================================
# 1. Distributed Setup
# ==========================================
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# ==========================================
# 2. Dataset with Strict Response Masking
# ==========================================
class LIMADataset(Dataset):
    def __init__(self, block_size=1024):
        self.dataset = load_dataset("GAIR/lima", split="train")
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.IGNORE_INDEX = -100 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        user_text = item['conversations'][0]
        assistant_text = item['conversations'][1]
        
        # ChatML Formatting
        prompt_str = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        response_str = f"{assistant_text}<|im_end|>"
        
        prompt_tokens = self.enc.encode(prompt_str, allowed_special="all")
        response_tokens = self.enc.encode(response_str, allowed_special="all")
        
        full_tokens = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        
        # Truncate
        if len(full_tokens) > self.block_size:
            full_tokens = full_tokens[:self.block_size]
            
        x = torch.tensor(full_tokens[:-1], dtype=torch.long)
        y = torch.tensor(full_tokens[1:], dtype=torch.long)
        
        # ✅ FIX 1: Response Masking (Ignore prompt in loss)
        y[:prompt_len - 1] = self.IGNORE_INDEX
        
        # ✅ FIX 2: Padding Masking
        actual_len = len(x)
        if actual_len < (self.block_size - 1):
            padding_len = (self.block_size - 1) - actual_len
            x = torch.cat([x, torch.full((padding_len,), self.enc.eot_token, dtype=torch.long)])
            y = torch.cat([y, torch.full((padding_len,), self.IGNORE_INDEX, dtype=torch.long)])
            
        return x, y

# ==========================================
# 3. Main Training Loop
# ==========================================
def main():
    local_rank = setup_ddp()
    master_process = (local_rank == 0)
    
    # Model Config
    config = GPTConfig()
    model = GPT(config)
    
    # Load Pre-trained weights (381k step checkpoint)
    checkpoint_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        # Remove 'module.' prefix if present from previous DDP training
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if master_process:
            print(f"✅ Loaded base model from {checkpoint_path}")

    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Dataset & Loader
    dataset = LIMADataset(block_size=config.block_size)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler) # 4 per GPU = 8 total
    
    # Optimizer (Low LR for SFT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
    
    model.train()
    if master_process:
        print("🚀 Starting SFT on Dual T4 GPUs...")

    for epoch in range(3): # LIMA usually needs 3-5 epochs
        sampler.set_epoch(epoch)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(local_rank), y.to(local_rank)
            
            optimizer.zero_grad()
            
            # Use Mixed Precision for speed on T4
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 5 == 0 and master_process:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    # Save final aligned model
    if master_process:
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "config": config
        }, "/kaggle/working/gpt250m_lima_sft_final.pth")
        print("🎉 SFT Completed! Model saved to /kaggle/working/")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()