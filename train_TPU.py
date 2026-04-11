import os
import time
import torch
import torch.nn.functional as F
import tiktoken

# XLA / TPU specific imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import our custom modules
from config import GPTConfig
from dataset import GPTDataset
from model import GPT

# ==========================================
# 1. Batch Generator (Must be accessible by spawn)
# ==========================================
def create_batch_generator(dataset, micro_batch_size, rank, start_step, seed=42):
    g = torch.Generator()
    # Unique seed per rank + step ensures no overlap/repetition on resume
    g.manual_seed(seed + rank + start_step) 
    max_idx = len(dataset)
    while True:
        x_batch, y_batch = [], []
        indices = torch.randint(0, max_idx, (micro_batch_size,), generator=g).tolist()
        for idx in indices:
            x, y = dataset[idx]
            x_batch.append(x)
            y_batch.append(y)
        yield torch.stack(x_batch), torch.stack(y_batch)

# ==========================================
# 2. Main Training Function
# ==========================================
def train_fn(index, config):
    """
    index: The specific core ID (0-7) assigned by xmp.spawn
    config: Passed from the main block
    """
    # TPU Setup
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    master_process = xm.is_master_ordinal()

    if master_process:
        print(f"🚀 Training initiated on TPU VM | Cores: {world_size}")

    # Configuration and Data Pipeline
    target_batch_size = 36
    micro_batch_size = 6
    gradient_accumulation_steps = target_batch_size // (micro_batch_size * world_size)

    data_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"
    
    if master_process:
        print("Loading Dataset...")
    train_dataset = GPTDataset(data_path=data_path, block_size=config.block_size)

    # Model Setup
    model = GPT(config).to(device)

    # Optimization Setup
    start_step = 0
    max_steps = 610352
    learning_rate = 5e-4
    warmup_steps = 10000

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps), eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Checkpointing Logic
    drive_path = "/kaggle/working/checkpoints"
    if master_process:
        os.makedirs(drive_path, exist_ok=True)

    working_checkpoint = os.path.join(drive_path, "latest_step_model.pth")
    input_checkpoint = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
    
    load_path = None
    if os.path.exists(working_checkpoint):
        load_path = working_checkpoint      
    elif os.path.exists(input_checkpoint):
        load_path = input_checkpoint        

    best_loss = float('inf')

    if load_path:
        # Load on CPU first to avoid OOM on TPU cores during loading
        checkpoint = torch.load(load_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        start_step = checkpoint["step"] + 1
        best_loss = checkpoint["best_loss"]
        
        if master_process:
            print(f"✅ Successfully resumed! Starting from step {start_step}")
        del checkpoint

    # Initialize generator (Fresh start or Resumed)
    train_iter = create_batch_generator(train_dataset, micro_batch_size, rank, start_step)

    # Training Loop
    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    for step in range(start_step, max_steps):
        
        # Get Batch
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()
        
        # Step Update (Gradient Accumulation)
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # xm.optimizer_step coordinates all TPU cores
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            scheduler.step()

        # Logging (Master process only)
        if master_process and step % 10 == 0:
            real_loss = loss.item() * gradient_accumulation_steps
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = (10 * target_batch_size * config.block_size) / dt
            print(f"Step {step:6d} | Loss: {real_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Speed: {tokens_per_sec:.2f} tok/s")
            t0 = time.time()

        # Checkpoint Saving
        if master_process and (step + 1) % 1000 == 0:
            checkpoint_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss
            }
            save_path = os.path.join(drive_path, "latest_step_model.pth")
            xm.save(checkpoint_data, save_path) # xm.save is preferred for TPU checkpoints
            print(f"💾 Checkpoint saved at step {step}")

# ==========================================
# 3. Entry Point
# ==========================================
if __name__ == '__main__':
    # Initialize the config here
    gpt_config = GPTConfig()
    
    # xmp.spawn using 'spawn' method (explicitly provided via nprocs)
    # nprocs=8 for Kaggle TPU v3-8
    xmp.spawn(train_fn, args=(gpt_config,), nprocs=8, start_method='spawn')