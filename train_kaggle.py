import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import tiktoken

from config import GPTConfig
from dataset import GPTDataset
from model import GPT

# ==========================================
# 🔥 0. Performance Boost Settings
# ==========================================
torch.backends.cudnn.benchmark = True

# ==========================================
# 1. Move Dataset to FAST STORAGE
# ==========================================
if not os.path.exists("/kaggle/working/train.bin"):
    print("Copying dataset to fast storage...")
    os.system("cp /kaggle/input/datasets/kundan8918/trainbin/train.bin /kaggle/working/")

DATA_PATH = "/kaggle/working/train.bin"

# ==========================================
# 2. Configuration
# ==========================================
config = GPTConfig()

target_batch_size = 128   # 🔥 increased
micro_batch_size = 8     # 🔥 increased

gradient_accumulation_steps = target_batch_size // micro_batch_size

print("Initializing dataset...")
train_dataset = GPTDataset(
    data_path=DATA_PATH,
    block_size=config.block_size
)

train_loader = DataLoader(
    train_dataset,
    batch_size=micro_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4,   # 🔥 increased
    persistent_workers=True
)

# ==========================================
# 3. Device + Model
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = GPT(config)
model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"🔥 Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# ==========================================
# 4. Optimization
# ==========================================
start_step = 0
max_steps = 610352

scaler = torch.amp.GradScaler('cuda')

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.95)
)

warmup_steps = 10000

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=(max_steps - warmup_steps),
    eta_min=1e-5
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

# ==========================================
# 5. Training Loop
# ==========================================
model.train()
optimizer.zero_grad(set_to_none=True)

t0 = time.time()

print("Starting training loop...")

train_iter = iter(train_loader)

for step in range(start_step, max_steps):

    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # 🔥 Mixed Precision
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss = loss / gradient_accumulation_steps

    scaler.scale(loss).backward()

    if (step + 1) % gradient_accumulation_steps == 0:

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    # ==========================================
    # Logging
    # ==========================================
    if step % 10 == 0 and step > 0:
        t1 = time.time()
        dt = t1 - t0

        tokens = 10 * micro_batch_size * config.block_size

        print(f"Step {step} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | Speed: {tokens/dt:.2f} tok/sec")

        t0 = time.time()

    # ==========================================
    # Save checkpoint (IMPORTANT FIX)
    # ==========================================
    if step % 1000 == 0 and step > 0:

        save_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        torch.save({
            "model": save_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step
        }, "/kaggle/working/model.pt")

        print("💾 Model saved!")

print("Training complete!")