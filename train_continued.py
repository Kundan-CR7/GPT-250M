"""
Continued pretraining on fineweb-edu starting from the existing 381k-step checkpoint.
Targets +10B tokens (~814k extra steps) → new max_steps = 1,195,000.
Keeps existing weights, optimizer momentum, and scaler state.
Builds a fresh LR schedule: 500-step bridge warmup + cosine to 1e-5 over the new horizon.
"""
import os, time, gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset
import tiktoken

from config import GPTConfig
from model import GPT

# =========================================================
# PATHS — EDIT THESE
# =========================================================
FINEWEB_PATH      = "/kaggle/input/YOUR-FINEWEB-DATASET/train.bin"    # <-- edit
OLD_CHECKPOINT    = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
WORKING_CKPT_DIR  = "/kaggle/working/checkpoints"

# =========================================================
# DDP SETUP
# =========================================================
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

if master_process:
    print(f"Device: {device} | GPUs: {ddp_world_size}")

# =========================================================
# CONFIG
# =========================================================
config = GPTConfig()
target_batch_size = 36
micro_batch_size  = 6
assert target_batch_size % (micro_batch_size * ddp_world_size) == 0
grad_accum_steps  = target_batch_size // (micro_batch_size * ddp_world_size)

max_steps     = 1_195_000          # 381k current + 814k new = +10B tokens
save_every    = 1000
bridge_warmup = 500                # smooths data-distribution shift
eta_min       = 1e-5

# =========================================================
# DATASET — memmap reader for a single train.bin
# =========================================================
class BinDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if master_process:
            print(f"Loaded {bin_path}: {len(self.data):,} tokens "
                  f"({len(self.data)*2/1e9:.2f} GB)")
    def __len__(self):
        return (len(self.data) - 1) // self.block_size
    def __getitem__(self, idx):
        start = np.random.randint(0, len(self.data) - self.block_size - 1)
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])

train_dataset = BinDataset(FINEWEB_PATH, config.block_size)

# =========================================================
# MODEL
# =========================================================
model = GPT(config).to(device)

# =========================================================
# OPTIMIZER + SCALER (built before loading checkpoint)
# =========================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95))
scaler    = torch.amp.GradScaler('cuda')

# =========================================================
# CHECKPOINT LOADING
# =========================================================
if master_process:
    os.makedirs(WORKING_CKPT_DIR, exist_ok=True)

working_ckpt = os.path.join(WORKING_CKPT_DIR, "latest_step_model.pth")
load_path = working_ckpt if os.path.exists(working_ckpt) else (
    OLD_CHECKPOINT if os.path.exists(OLD_CHECKPOINT) else None)

start_step = 0
best_loss  = float('inf')

assert load_path, "No checkpoint to resume from."
if master_process:
    print(f"Loading checkpoint: {load_path}")

ckpt = torch.load(load_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
scaler.load_state_dict(ckpt["scaler_state_dict"])
start_step = ckpt["step"] + 1
best_loss  = ckpt["best_loss"]
# Note: intentionally NOT loading old scheduler state.

del ckpt; gc.collect(); torch.cuda.empty_cache()

if master_process:
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"Resumed at step {start_step} | current LR {cur_lr:.2e} | best_loss {best_loss:.4f}")

# =========================================================
# FRESH SCHEDULER — new horizon, bridge warmup + cosine
# =========================================================
remaining_steps = max(max_steps - start_step, bridge_warmup + 100)
warmup_sched = LinearLR(optimizer, start_factor=0.3, end_factor=1.0,
                        total_iters=bridge_warmup)
cosine_sched = CosineAnnealingLR(optimizer,
                                 T_max=remaining_steps - bridge_warmup,
                                 eta_min=eta_min)
scheduler = SequentialLR(optimizer,
                         schedulers=[warmup_sched, cosine_sched],
                         milestones=[bridge_warmup])

if master_process:
    print(f"New schedule: {bridge_warmup}-step bridge warmup, "
          f"then cosine over {remaining_steps - bridge_warmup} steps to eta_min={eta_min}")

# =========================================================
# DDP WRAP
# =========================================================
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# =========================================================
# SAMPLING HELPER
# =========================================================
@torch.no_grad()
def generate_sample(m, dev, prompt="Tell me something about AI", max_new=70):
    m.eval()
    enc = tiktoken.get_encoding("gpt2")
    x = torch.tensor(enc.encode(prompt), dtype=torch.long, device=dev).unsqueeze(0)
    print(f"\n--- Brain check ---\nPrompt: {prompt}")
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for _ in range(max_new):
            x_cond = x[:, -config.block_size:]
            logits = m(x_cond)[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat((x, nxt), dim=1)
    print(f"Output: {enc.decode(x[0].tolist())}\n-------------------\n")
    m.train()
    gc.collect(); torch.cuda.empty_cache()

# =========================================================
# BATCH GENERATOR
# =========================================================
def batch_gen(dataset, micro_bs, rank, start_step, seed=42):
    g = torch.Generator(); g.manual_seed(seed + rank + start_step)
    N = len(dataset)
    while True:
        idxs = torch.randint(0, N, (micro_bs,), generator=g).tolist()
        xs, ys = [], []
        for i in idxs:
            x, y = dataset[i]; xs.append(x); ys.append(y)
        yield torch.stack(xs), torch.stack(ys)

train_iter = batch_gen(train_dataset, micro_batch_size, ddp_rank, start_step)

# =========================================================
# TRAIN LOOP
# =========================================================
model.train()
optimizer.zero_grad(set_to_none=True)
t0 = time.time()

if master_process:
    print(f"Training from step {start_step} to {max_steps} "
          f"(+{max_steps - start_step:,} steps ≈ "
          f"{(max_steps - start_step) * micro_batch_size * ddp_world_size * config.block_size / 1e9:.2f}B tokens)")

for step in range(start_step, max_steps):
    x, y = next(train_iter)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    sync_now = (step + 1) % grad_accum_steps == 0
    if ddp: model.require_backward_grad_sync = sync_now

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss = loss / grad_accum_steps
    scaler.scale(loss).backward()

    if sync_now:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    if master_process:
        real_loss = loss.item() * grad_accum_steps

        if step % 10 == 0 and step > start_step:
            dt = time.time() - t0
            toks = 10 * micro_batch_size * ddp_world_size * config.block_size
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"step {step:7d} | loss {real_loss:.4f} | "
                  f"lr {lr_now:.2e} | {toks/dt:,.0f} tok/s")
            t0 = time.time()

        if sync_now:
            ck = {
                "step": step,
                "model_state_dict":     raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            }
            if step >= 400 and real_loss < best_loss and real_loss < 4.5:
                best_loss = real_loss
                ck["best_loss"] = best_loss
                torch.save(ck, os.path.join(WORKING_CKPT_DIR, "best_model.pth"))
                print(f"New best loss {best_loss:.4f}")
            if (step + 1) % save_every == 0:
                torch.save(ck, os.path.join(WORKING_CKPT_DIR, "latest_step_model.pth"))
                print(f"Saved latest checkpoint at step {step}")
                generate_sample(raw_model, device)

if ddp:
    dist.destroy_process_group()
if master_process:
    print("Run complete.")
