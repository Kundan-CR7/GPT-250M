import os, time, gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset

from config import GPTConfig
from model import GPT

# =========================================================
# PATHS
# =========================================================
FINEWEB_PATH     = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"
OLD_CHECKPOINT   = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
WORKING_CKPT_DIR = "/kaggle/working/checkpoints"

# =========================================================
# DDP SETUP
# =========================================================
ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank       = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

device_type = "cuda" if "cuda" in device else "cpu"

if master_process:
    print(f"Device: {device} | GPUs: {ddp_world_size}")

# =========================================================
# CONFIG
# =========================================================
config = GPTConfig()

target_batch_size = 36
micro_batch_size  = 6

base_lr       = 5e-4
eta_min       = 1e-5
max_steps     = 1_195_000
save_every    = 3000
bridge_warmup = 500

assert target_batch_size % (micro_batch_size * ddp_world_size) == 0
grad_accum_steps = target_batch_size // (micro_batch_size * ddp_world_size)

# =========================================================
# DATASET
# =========================================================
class BinDataset(Dataset):
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.max_start = len(self.data) - block_size - 1

        if master_process:
            print(f"Loaded dataset: {len(self.data):,} tokens")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        start = idx % self.max_start
        chunk = self.data[start:start+self.block_size+1].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

train_dataset = BinDataset(FINEWEB_PATH, config.block_size)

# =========================================================
# MODEL
# =========================================================
model = GPT(config).to(device)

# =========================================================
# OPTIMIZER + AMP
# =========================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=base_lr,
    betas=(0.9, 0.95)
)

scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

# =========================================================
# CHECKPOINT LOAD
# =========================================================
os.makedirs(WORKING_CKPT_DIR, exist_ok=True)
working_ckpt = os.path.join(WORKING_CKPT_DIR, "latest_step_model.pth")

is_continuation = os.path.exists(working_ckpt)

load_path = (
    working_ckpt if is_continuation
    else OLD_CHECKPOINT if os.path.exists(OLD_CHECKPOINT)
    else None
)

start_step = 0
opt_step = 0
best_loss = float("inf")
scheduler_state = None

if load_path is not None:

    if master_process:
        print(f"Loading checkpoint: {load_path}")

    ckpt = torch.load(load_path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_step = ckpt.get("step", -1) + 1
    opt_step   = ckpt.get("opt_step", start_step // grad_accum_steps)

    best_loss = ckpt.get("best_loss", float("inf"))
    scheduler_state = ckpt.get("scheduler_state_dict", None)

    # NEW SESSION from old pretrained checkpoint
    if not is_continuation:
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr
            pg["initial_lr"] = base_lr

        scheduler_state = None
        opt_step = 0

        if master_process:
            print("Fresh session detected -> LR reset to 5e-4")

    del ckpt
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if master_process:
    print(
        f"Resume step={start_step} | "
        f"opt_step={opt_step} | "
        f"lr={optimizer.param_groups[0]['lr']:.4e}"
    )

# =========================================================
# ALWAYS SAME SCHEDULER ARCHITECTURE
# =========================================================
total_updates  = max_steps // grad_accum_steps
warmup_updates = max(1, bridge_warmup // grad_accum_steps)
cosine_updates = max(1, total_updates - warmup_updates)

warmup_sched = LinearLR(
    optimizer,
    start_factor=0.3,
    end_factor=1.0,
    total_iters=warmup_updates
)

cosine_sched = CosineAnnealingLR(
    optimizer,
    T_max=cosine_updates,
    eta_min=eta_min
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_sched, cosine_sched],
    milestones=[warmup_updates]
)

if scheduler_state is not None:
    try:
        scheduler.load_state_dict(scheduler_state)
        if master_process:
            print("Scheduler restored successfully.")
    except Exception as e:
        if master_process:
            print(f"Scheduler mismatch -> fresh scheduler. {e}")

# =========================================================
# DDP WRAP
# =========================================================
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# =========================================================
# DDP SAFE RANDOM BATCH GENERATOR
# =========================================================
def batch_gen(dataset, micro_bs, rank, step_seed):
    g = torch.Generator()
    g.manual_seed(42 + rank + step_seed)

    N = len(dataset)

    while True:
        idxs = torch.randint(0, N, (micro_bs,), generator=g)
        xs, ys = zip(*[dataset[i.item()] for i in idxs])

        yield torch.stack(xs), torch.stack(ys)

train_iter = batch_gen(train_dataset, micro_batch_size, ddp_rank, start_step)

# =========================================================
# TRAIN LOOP
# =========================================================
model.train()
optimizer.zero_grad(set_to_none=True)

running_loss = 0.0
t0 = time.time()

for step in range(start_step, max_steps):

    x, y = next(train_iter)

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    sync_now = ((step + 1) % grad_accum_steps == 0)

    if ddp:
        model.require_backward_grad_sync = sync_now

    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16 if device_type == "cuda" else torch.float32
    ):
        logits = model(x)

        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B*T, C),
            y.reshape(B*T)
        )

        running_loss += loss.detach().item()

        loss = loss / grad_accum_steps

    scaler.scale(loss).backward()

    if sync_now:

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        opt_step += 1

        avg_loss = running_loss / grad_accum_steps
        running_loss = 0.0

        # LOGGING
        if master_process and opt_step % 10 == 0:

            dt = time.time() - t0
            toks = 10 * target_batch_size * config.block_size
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"step {step} | "
                f"opt_step {opt_step} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.4e} | "
                f"{toks/dt:,.0f} tok/s"
            )

            t0 = time.time()

        # SAVE
        if master_process:

            ckpt = {
                "step": step,
                "opt_step": opt_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            }

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    ckpt,
                    os.path.join(WORKING_CKPT_DIR, "best_model.pth")
                )

            if (step + 1) % save_every == 0:
                torch.save(
                    ckpt,
                    os.path.join(WORKING_CKPT_DIR, "latest_step_model.pth")
                )
                print(f"Checkpoint saved at step {step}")

# =========================================================
# CLEANUP
# =========================================================
if ddp:
    dist.destroy_process_group()

if master_process:
    print("Training complete.")