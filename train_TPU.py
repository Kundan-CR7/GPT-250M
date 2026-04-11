import os

# ✅ CRITICAL: Must be set before ANY torch_xla import
os.environ['CLOUD_TPU_TASK_ID'] = '0'
os.environ['TPU_PROCESS_ADDRESSES'] = 'localhost:8476,localhost:8477,localhost:8478,localhost:8479,localhost:8480,localhost:8481,localhost:8482,localhost:8483'
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '2,2,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['XLA_USE_BF16'] = '1'

import time
import torch
import torch.nn.functional as F

# XLA / TPU specific imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import our custom modules
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
    # TPU Setup
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    master_process = xm.is_master_ordinal()

    if master_process:
        print(f"🚀 Training initiated on TPU VM | Cores: {world_size}")

    # ── Batch size config ──────────────────────────────────────────────────────
    # With 8 TPU cores and micro_batch=6, each optimizer step sees:
    # 6 * 8 * grad_accum = target_batch_size tokens worth of sequences
    target_batch_size = 480          # 6 * 8 * 10 — one clean optimizer step
    micro_batch_size  = 6
    gradient_accumulation_steps = target_batch_size // (micro_batch_size * world_size)
    # = 480 // (6 * 8) = 10
    # ──────────────────────────────────────────────────────────────────────────

    data_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"

    if master_process:
        print("Loading Dataset...")
    train_dataset = GPTDataset(data_path=data_path, block_size=config.block_size)

    # Model
    model = GPT(config).to(device)

    # Optimiser & schedulers
    start_step    = 0
    max_steps     = 610352
    learning_rate = 5e-4
    warmup_steps  = 10000

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95)
    )
    warmup_scheduler  = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler  = CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps), eta_min=1e-5
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ── Checkpointing ─────────────────────────────────────────────────────────
    drive_path = "/kaggle/working/checkpoints"
    os.makedirs(drive_path, exist_ok=True)   # safe on every rank

    working_checkpoint = os.path.join(drive_path, "latest_step_model.pth")
    input_checkpoint   = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"

    load_path = None
    if os.path.exists(working_checkpoint):
        load_path = working_checkpoint
    elif os.path.exists(input_checkpoint):
        load_path = input_checkpoint

    best_loss = float("inf")

    if load_path:
        # Load on CPU first to avoid OOM on TPU HBM
        checkpoint = torch.load(load_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_step = checkpoint["step"] + 1
        best_loss  = checkpoint.get("best_loss", float("inf"))

        if master_process:
            print(f"✅ Resumed from step {start_step}")
        del checkpoint
    # ──────────────────────────────────────────────────────────────────────────

    train_iter = create_batch_generator(
        train_dataset, micro_batch_size, rank, start_step
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    for step in range(start_step, max_steps):

        # ── Forward / backward ────────────────────────────────────────────────
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # ── Optimizer step (every `gradient_accumulation_steps` micro-steps) ──
        # Use position within the accumulation window, not raw step index,
        # so resuming mid-window doesn't skip or double-apply an update.
        steps_since_start = step - start_step + 1
        if steps_since_start % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)   # syncs all 8 cores automatically
            optimizer.zero_grad()
            scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if master_process and step % 10 == 0:
            # loss.item() forces a device→host sync; batch with the log print
            real_loss = loss.item() * gradient_accumulation_steps
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = (10 * target_batch_size * config.block_size) / dt
            print(
                f"Step {step:6d} | Loss: {real_loss:.4f} "
                f"| LR: {scheduler.get_last_lr()[0]:.2e} "
                f"| Speed: {tokens_per_sec:.0f} tok/s"
            )
            t0 = time.time()

        # ── Checkpointing ─────────────────────────────────────────────────────
        if (step + 1) % 1000 == 0:
            # xm.rendezvous ensures all cores finish their current work
            # before any core starts writing to disk.
            xm.rendezvous("checkpoint_sync")

            current_loss = loss.item() * gradient_accumulation_steps
            if current_loss < best_loss:
                best_loss = current_loss

            checkpoint_data = {
                "step":                step,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss":           best_loss,
            }
            save_path = os.path.join(drive_path, "latest_step_model.pth")

            # xm.save must be called by ALL ranks — it handles master-only
            # writing internally.  Do NOT wrap in `if master_process`.
            xm.save(checkpoint_data, save_path)

            if master_process:
                print(f"💾 Checkpoint saved at step {step} | best_loss: {best_loss:.4f}")


# ==========================================
# 3. Entry Point
# ==========================================
if __name__ == "__main__":
    gpt_config = GPTConfig()

    # ✅ No XLA calls here — let xmp.spawn initialise each core cleanly.
    # nprocs=None  →  PJRT auto-detects all 8 TPU cores.
    xmp.spawn(train_fn, args=(gpt_config,), start_method="spawn")