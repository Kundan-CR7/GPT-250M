import os
import requests

# ==========================================
# TPU Environment Setup
# (MUST happen before any torch_xla import)
# ==========================================
def get_kaggle_tpu_address():
    url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/TPU_PROCESS_ADDRESSES"
    try:
        r = requests.get(url, headers={"Metadata-Flavor": "Google"}, timeout=5)
        return r.text
    except Exception:
        return None

tpu_addr = get_kaggle_tpu_address()
print(f"DEBUG TPU_PROCESS_ADDRESSES = '{tpu_addr}'")

if tpu_addr and len(tpu_addr.split(',')) == 8:
    os.environ['TPU_PROCESS_ADDRESSES'] = tpu_addr
    print(f"✅ TPU addresses fetched from metadata: {tpu_addr}")
else:
    # Fallback: hardcoded localhost ports for Kaggle TPU v3-8
    fallback = (
        'localhost:8476,localhost:8477,localhost:8478,localhost:8479,'
        'localhost:8480,localhost:8481,localhost:8482,localhost:8483'
    )
    os.environ['TPU_PROCESS_ADDRESSES'] = fallback
    print(f"⚠️  Using fallback TPU addresses: {fallback}")

os.environ['CLOUD_TPU_TASK_ID']            = '0'
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '2,2,1'
os.environ['TPU_PROCESS_BOUNDS']           = '1,1,1'
os.environ['PJRT_DEVICE']                  = 'TPU'

# BF16: XLA_USE_BF16 is deprecated in torch_xla 2.6+
# We will cast the model manually instead (done below in train_fn)

import time
import torch
import torch.nn.functional as F

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import GPTConfig
from dataset import GPTDataset
from model import GPT


# ==========================================
# 1. Batch Generator
# ==========================================
def create_batch_generator(dataset, micro_batch_size, rank, start_step, seed=42):
    g = torch.Generator()
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

    # ── TPU / distributed setup ───────────────────────────────────────────────
    device       = xm.xla_device()
    world_size   = xr.world_size()
    rank         = xr.global_ordinal()
    master_process = (rank == 0)

    if master_process:
        print(f"🚀 Training started | Cores: {world_size} | Device: {device}")

    # ── Batch / accumulation config ───────────────────────────────────────────
    micro_batch_size          = 6
    target_batch_size         = micro_batch_size * world_size * 10   # 10 accum steps
    gradient_accumulation_steps = target_batch_size // (micro_batch_size * world_size)
    # = 10 regardless of world_size

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"
    if master_process:
        print("Loading dataset...")
    train_dataset = GPTDataset(data_path=data_path, block_size=config.block_size)

    # ── Model (cast to bf16 manually — XLA_USE_BF16 is deprecated) ───────────
    model = GPT(config).to(torch.bfloat16).to(device)

    # ── Optimiser & schedulers ────────────────────────────────────────────────
    start_step    = 0
    max_steps     = 610352
    learning_rate = 5e-4
    warmup_steps  = 10000

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95)
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps), eta_min=1e-5
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ── Checkpoint loading ────────────────────────────────────────────────────
    drive_path         = "/kaggle/working/checkpoints"
    os.makedirs(drive_path, exist_ok=True)

    working_checkpoint = os.path.join(drive_path, "latest_step_model.pth")
    input_checkpoint   = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"

    load_path = None
    if os.path.exists(working_checkpoint):
        load_path = working_checkpoint
    elif os.path.exists(input_checkpoint):
        load_path = input_checkpoint

    best_loss = float("inf")

    if load_path:
        if master_process:
            print(f"📂 Loading checkpoint from: {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        # Optimizer state may be incompatible if model architecture changed
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as e:
            if master_process:
                print(f"⚠️  Optimizer state skipped (architecture changed): {e}")
                print("   Continuing with a fresh optimizer.")

        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                if master_process:
                    print(f"⚠️  Scheduler state skipped: {e}")

        start_step = checkpoint["step"] + 1
        best_loss  = checkpoint.get("best_loss", float("inf"))

        if master_process:
            print(f"✅ Resumed from step {start_step} | best_loss so far: {best_loss:.4f}")

        del checkpoint

    # ── Data iterator ─────────────────────────────────────────────────────────
    train_iter = create_batch_generator(
        train_dataset, micro_batch_size, rank, start_step
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    for step in range(start_step, max_steps):

        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Forward
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss = loss / gradient_accumulation_steps

        # Backward
        loss.backward()

        # Optimizer step — keyed on position within the accumulation window
        # so resuming mid-window never skips or double-applies an update
        steps_since_start = step - start_step + 1
        if steps_since_start % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)   # syncs all TPU cores
            optimizer.zero_grad()
            scheduler.step()

        # Logging (master only, every 10 steps)
        if master_process and step % 10 == 0:
            real_loss = loss.item() * gradient_accumulation_steps
            t1 = time.time()
            tokens_per_sec = (10 * target_batch_size * config.block_size) / (t1 - t0)
            print(
                f"Step {step:6d} | Loss: {real_loss:.4f} "
                f"| LR: {scheduler.get_last_lr()[0]:.2e} "
                f"| Speed: {tokens_per_sec:.0f} tok/s"
            )
            t0 = time.time()

        # Checkpoint (all ranks must call xm.save)
        if (step + 1) % 1000 == 0:
            xm.rendezvous("checkpoint_sync")   # wait for all cores to finish

            current_loss = loss.item() * gradient_accumulation_steps
            if current_loss < best_loss:
                best_loss = current_loss

            checkpoint_data = {
                "step":                 step,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss":            best_loss,
            }
            save_path = os.path.join(drive_path, "latest_step_model.pth")

            # xm.save must be called by ALL ranks — do NOT gate on master_process
            xm.save(checkpoint_data, save_path)

            if master_process:
                print(f"💾 Checkpoint saved at step {step} | best_loss: {best_loss:.4f}")


# ==========================================
# 3. Entry Point
# ==========================================
if __name__ == "__main__":
    gpt_config = GPTConfig()

    # No XLA calls here — xmp.spawn initialises each core cleanly
    xmp.spawn(train_fn, args=(gpt_config,), start_method="spawn")