import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import tiktoken

# Import our custom modules (Ensure these files are also uploaded/written to Kaggle working dir)
from config import GPTConfig
from dataset import GPTDataset
from model import GPT

# ==========================================
# 1. DDP Initialization & Hardware Setup
# ==========================================
# Check if we are running under torchrun
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # Only GPU 0 prints, saves, and evaluates
else:
    # Fallback for single GPU
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

if master_process:
    print(f"Training on device: {device} | GPUs detected: {ddp_world_size}")

# ==========================================
# 2. Configuration and Data Pipeline
# ==========================================
config = GPTConfig()

# ⚡ DDP MATH: Target is 32. 
# 2 GPUs * 8 micro_batch = 16 sequences per forward pass.
# 32 / 16 = 2 gradient accumulation steps (Twice as fast as Colab!)
target_batch_size = 36
micro_batch_size = 6
assert target_batch_size % (micro_batch_size * ddp_world_size) == 0
gradient_accumulation_steps = target_batch_size // (micro_batch_size * ddp_world_size)

# Update this path to your Kaggle dataset input!
data_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"

if master_process:
    print("Initializing dataset...")
train_dataset = GPTDataset(data_path=data_path, block_size=config.block_size)

# ==========================================
# 3. Model Setup & DDP Wrapping
# ==========================================
model = GPT(config)
model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
# When saving checkpoints, we must pull the raw model out of the DDP wrapper
raw_model = model.module if ddp else model

# ==========================================
# 4. Optimization Setup
# ==========================================
start_step = 0
max_steps = 610352
save_interval = 500

scaler = torch.amp.GradScaler('cuda')
learning_rate = 5e-4
warmup_steps = 10000

optimizer = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(max_steps-warmup_steps), eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# ==========================================
# 5. Checkpointing Setup (Kaggle)
# ==========================================
drive_path = "/kaggle/working/checkpoints"
if master_process:
    os.makedirs(drive_path, exist_ok=True)

# 1. Where we save NEW checkpoints (Read/Write)
working_checkpoint = os.path.join(drive_path, "latest_step_model.pth")

# 2. Where we load your UPLOADED checkpoint from (Read-Only)
# ⚠️ This path must exactly match where your uploaded dataset is!
input_checkpoint = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"

# 3. Determine which file to load
load_path = None
if os.path.exists(working_checkpoint):
    load_path = working_checkpoint      # Resume from a crash in THIS Kaggle session
elif os.path.exists(input_checkpoint):
    load_path = input_checkpoint        # Resume from your uploaded Google Drive checkpoint

best_loss = float('inf')

if load_path:
    if master_process:
        print(f"Loading checkpoint from: {load_path}...")
    checkpoint = torch.load(load_path, map_location="cpu")
    
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_step = checkpoint["step"] + 1
    best_loss = checkpoint["best_loss"]

    del checkpoint
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    if master_process:
        print(f"Successfully resumed! Starting from step {start_step}")
else:
    if master_process:
        print("No checkpoint found. Starting training from scratch!")

# ==========================================
# 6. Evaluation Sampling Function
# ==========================================
def generate_sample(model, device, prompt="The ", max_new_tokens=30):
    model.eval()
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n--- 🧠 Model Brain Check ---")
    print(f"Prompt: '{prompt}'")
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(max_new_tokens):
                x_cond = x[:, -config.block_size:]
                logits = model(x_cond)
                logits = logits[:, -1, :] 
                probs = F.softmax(logits / 0.8, dim=-1) 
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
            
    output_text = enc.decode(x[0].tolist())
    print(f"Output: {output_text}")
    print("----------------------------\n")
    model.train()
    del x, tokens, logits, probs, next_token
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 7. The Distributed Training Loop
# ==========================================
model.train()

# Set up the DataLoader with DistributedSampler for DDP
if start_step > 0:
    # Calculate global sequences consumed across ALL GPUs
    samples_to_skip = start_step * (micro_batch_size * ddp_world_size)
    if master_process:
        print(f"⏩ O(1) Resume: Jumping directly to sequence index {samples_to_skip}...")
    
    remaining_indices = range(samples_to_skip, len(train_dataset))
    resumed_dataset = Subset(train_dataset, remaining_indices)
    
    sampler = DistributedSampler(resumed_dataset, shuffle=False) if ddp else None
    train_loader = DataLoader(resumed_dataset, batch_size=micro_batch_size, sampler=sampler, shuffle=False, pin_memory=True, num_workers=0)
else:
    sampler = DistributedSampler(train_dataset, shuffle=False) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, sampler=sampler, shuffle=False, pin_memory=True, num_workers=0)

train_iter = iter(train_loader)
optimizer.zero_grad(set_to_none=True)
t0 = time.time()

if master_process:
    print("Starting distributed training loop...")

for step in range(start_step, max_steps):
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
        
    x, y = x.to(device), y.to(device)

    # Disable gradient sync for accumulation steps to dramatically speed up DDP
    require_backward_grad_sync = (step + 1) % gradient_accumulation_steps == 0
    if ddp and not require_backward_grad_sync:
        model.require_backward_grad_sync = False
    else:
        if ddp: model.require_backward_grad_sync = True

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss = loss / gradient_accumulation_steps

    scaler.scale(loss).backward()
    
    if require_backward_grad_sync:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    # ==========================================
    # 8. Logging and Checkpoint Saving (Master Only)
    # ==========================================
    if master_process:
        # In DDP, this loss is technically only GPU 0's loss, but it's an accurate enough 
        # representation without slowing down the GPUs by forcing them to sync and average.
        real_loss = loss.item() * gradient_accumulation_steps
        
        if step % 10 == 0 and step > 0:
            t1 = time.time()
            dt = t1 - t0
            # Global tokens processed across all GPUs
            tokens_processed = 10 * (micro_batch_size * ddp_world_size) * config.block_size
            print(f"Step {step:5d} | Loss: {real_loss:.4f} | Speed: {(tokens_processed / dt):.2f} tok/sec")
            t0 = time.time() 

        if require_backward_grad_sync:
            checkpoint = {
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss if 'best_loss' in locals() else real_loss
            }
            if step >= 400 and real_loss < best_loss:
                best_loss = real_loss
                best_path = os.path.join(drive_path, "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"🌟 New Best Model! Saved to Kaggle Dir (Loss: {best_loss:.4f})")

            if (step + 1) % 1000 == 0:
                interval_path = os.path.join(drive_path, "latest_step_model.pth")
                torch.save(checkpoint, interval_path)
                print(f"💾 Interval Backup! Saved step {step} to Kaggle Dir.")
                generate_sample(raw_model, device)

if ddp:
    dist.destroy_process_group()
if master_process:
    print("Training run completed!")