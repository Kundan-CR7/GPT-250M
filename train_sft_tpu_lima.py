import os
os.environ['PJRT_DEVICE'] = 'TPU'

# Optional: Force the TPU to use all 8 devices
os.environ['TPU_NUM_DEVICES'] = '8'
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# PyTorch XLA Imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# Import your custom modules
from config import GPTConfig
from model import GPT

# ==========================================
# 1. FIXED: LIMA Dataset with Strict Masking
# ==========================================
class LIMADataset(Dataset):
    def __init__(self, block_size=1024):
        # We use 'GAIR/lima' but ensure we only take the train split
        self.dataset = load_dataset("GAIR/lima", split="train")
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.IGNORE_INDEX = -100 # Standard for CrossEntropy to ignore
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        user_text = item['conversations'][0]
        assistant_text = item['conversations'][1]
        
        # 1. Define the separate parts to find the split point
        prompt_str = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        response_str = f"{assistant_text}<|im_end|>"
        
        prompt_tokens = self.enc.encode(prompt_str, allowed_special="all")
        response_tokens = self.enc.encode(response_str, allowed_special="all")
        
        full_tokens = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        
        # 2. Truncate if over block size
        if len(full_tokens) > self.block_size:
            full_tokens = full_tokens[:self.block_size]
            
        # 3. Create X and Y
        # x is the input sequence
        # y is the target sequence (shifted by 1)
        x = torch.tensor(full_tokens[:-1], dtype=torch.long)
        y = torch.tensor(full_tokens[1:], dtype=torch.long)
        
        # 4. CRITICAL: Response Masking
        # Mask everything in Y that corresponds to the user's prompt
        # (prompt_len - 1) because Y is already shifted by 1
        y[:prompt_len - 1] = self.IGNORE_INDEX
        
        # 5. FIXED: Padding and Pad Masking
        actual_len = len(x)
        if actual_len < (self.block_size - 1):
            padding_len = (self.block_size - 1) - actual_len
            
            # Pad X with EOT
            x = torch.cat([x, torch.full((padding_len,), self.enc.eot_token, dtype=torch.long)])
            
            # Pad Y with IGNORE_INDEX so padding doesn't contribute to loss
            y = torch.cat([y, torch.full((padding_len,), self.IGNORE_INDEX, dtype=torch.long)])
            
        return x, y

# ==========================================
# 2. The TPU Training Function
# ==========================================
def train_loop_fn(index, flags):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    torch.manual_seed(flags['seed'])
    device = xm.xla_device()
    
    # Initialize Model
    config = GPTConfig()
    model = GPT(config)
    
    # Load GPU-trained weights on CPU first
    if os.path.exists(flags['checkpoint_path']):
        checkpoint = torch.load(flags['checkpoint_path'], map_location="cpu")
        # Handle potential DDP prefix if needed
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if xm.is_master_proc():
            print(f"✅ Successfully loaded base model from {flags['checkpoint_path']}")
    
    model.to(device)
    model.train()

    # Data Loading
    dataset = LIMADataset(block_size=config.block_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )
    train_loader = DataLoader(dataset, batch_size=flags['batch_size'], sampler=train_sampler)
    
    # TPU Parallel Loader
    tpu_loader = pl.MpDeviceLoader(train_loader, device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=flags['lr'], weight_decay=0.1)

    if xm.is_master_proc():
        print(f"🚀 Starting Masked SFT on {xm.xrt_world_size()} TPU Cores")

    for epoch in range(flags['epochs']):
        for step, (x, y) in enumerate(tpu_loader):
            optimizer.zero_grad()
            
            logits = model(x)
            # F.cross_entropy by default uses ignore_index=-100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            
            loss.backward()
            
            # TPU Gradient Clipping & Step
            xm.optimizer_step(optimizer)
            
            if step % 10 == 0 and xm.is_master_proc():
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # Save after every epoch
        if xm.is_master_proc():
            save_path = f"gpt250m_lima_sft_v2_epoch_{epoch}.pth"
            xm.save(model.state_dict(), save_path)

# ==========================================
# 3. Launcher
# ==========================================
FLAGS = {
    'checkpoint_path': '/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth',
    'lr': 1e-5,          # Lower LR for LIMA to maintain stability
    'batch_size': 4,     # Per-core batch size
    'epochs': 4,         # LIMA needs ~3-5 epochs
    'seed': 42
}

if __name__ == '__main__':
    # Set nprocs to None so it automatically uses all 8 TPU cores
    xmp.spawn(train_loop_fn, args=(FLAGS,), nprocs=None)