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
# 1. Dataset with Improved Formatting
# ==========================================
class AlpacaCorrectedDataset(Dataset):
    def __init__(self, block_size=1024, num_samples=10000):
        # Increased samples for better instruction coverage
        self.dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{num_samples}]")
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.IGNORE_INDEX = -100 
        
        # Ensure special tokens are handled (conceptual mapping)
        # Note: In gpt2 tiktoken, we use these strings directly. 
        # The model learns their meaning during SFT.
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        instruction = item['instruction']
        input_text = item['input']
        output_text = item['output']
        
        # Clean formatting
        user_prompt = f"{instruction}\n{input_text}".strip()
        
        # ChatML Formatting - The "Secret Sauce" for consistency
        prompt_str = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        response_str = f"{output_text}<|im_end|>"
        
        prompt_tokens = self.enc.encode(prompt_str, allowed_special="all")
        response_tokens = self.enc.encode(response_str, allowed_special="all")
        
        full_tokens = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        
        # Truncate if necessary
        if len(full_tokens) > self.block_size:
            full_tokens = full_tokens[:self.block_size]
            
        x = torch.tensor(full_tokens[:-1], dtype=torch.long)
        y = torch.tensor(full_tokens[1:], dtype=torch.long)
        
        # ✅ FIX 1: Strict Response Masking
        # We only want to calculate loss on the Assistant's words
        y[:prompt_len - 1] = self.IGNORE_INDEX
        
        # ✅ FIX 2: Better Padding Handling
        actual_len = len(x)
        if actual_len < (self.block_size - 1):
            padding_len = (self.block_size - 1) - actual_len
            # Pad X with EOT, Pad Y with IGNORE_INDEX
            x = torch.cat([x, torch.full((padding_len,), self.enc.eot_token, dtype=torch.long)])
            y = torch.cat([y, torch.full((padding_len,), self.IGNORE_INDEX, dtype=torch.long)])
            
        return x, y

# ==========================================
# 2. Training Loop with Stability Fixes
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig()
    model = GPT(config)
    
    # LOAD THE ORIGINAL 381K CHECKPOINT
    checkpoint_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state_dict)
        print("✅ Base Model Loaded. Starting specialized SFT.")

    model.to(device)
    
    dataset = AlpacaCorrectedDataset(block_size=config.block_size)
    # Batch size 4 + Grad Accumulation 4 = Effective Batch Size 32
    loader = DataLoader(dataset, batch_size=4, shuffle=True) 
    grad_accum_steps = 4 
    
    # ⚡ LR Tweak: 8e-6 (Slightly lower than 1e-5 for maximum safety)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-6, weight_decay=0.1)
    
    model.train()
    print(f"🚀 Training on {device}...")

    for epoch in range(2): # 2 Epochs is the sweet spot for 250M
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                loss = loss / grad_accum_steps # Scale loss for accumulation
            
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 20 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item() * grad_accum_steps:.4f}")
            
            # Frequent Checkpoints
            if step % 500 == 0 and step > 0:
                torch.save(model.state_dict(), f"/kaggle/working/gpt250m_sft_step_{step}.pth")

    torch.save(model.state_dict(), "/kaggle/working/gpt250m_sft_final_v2.pth")
    print("🎉 SFT Complete. Model is ready for FinOps-AI testing.")

if __name__ == "__main__":
    main()