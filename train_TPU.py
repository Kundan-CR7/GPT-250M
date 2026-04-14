import os
import time
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import tiktoken

# ==========================================
# TPU-SPECIFIC IMPORTS
# ==========================================
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
# BUG 1 FIX: This import registers the 'xla' backend string with torch.distributed.
# Without it, dist.init_process_group(backend='xla') raises RuntimeError: "no such backend".
import torch_xla.distributed.xla_backend  # <-- REQUIRED, not optional

from config import GPTConfig
from dataset import GPTDataset
from model import GPT


def train_fn(index):
    """
    All training logic lives inside this function.
    xmp.spawn() calls this once per TPU core (typically 8 cores on a v3/v4 TPU).
    `index` is the local rank (0-7).
    """

    # ==========================================
    # 1. TPU Device & Distributed Setup
    # ==========================================
    device = xm.xla_device()

    dist.init_process_group(
        backend='xla',        # Works now that xla_backend is imported above
        init_method='xla://',
    )

    ddp_rank       = xm.get_ordinal()          # global rank
    ddp_local_rank = index                     # local core index
    ddp_world_size = xm.xrt_world_size()       # total number of cores
    master_process  = xm.is_master_ordinal()   # True only on rank 0

    if master_process:
        print(f"Training on device: {device} | TPU cores: {ddp_world_size}")

    # ==========================================
    # 2. Configuration and Data Pipeline
    # ==========================================
    config = GPTConfig()

    # BUG 2 FIX: Original values (target=36, micro=6) fail the assert on 8 cores:
    #   36 % (6 * 8) == 36 % 48 == 36  !=  0  →  AssertionError at startup.
    # Fix: scale up so target_batch_size is divisible by (micro_batch_size * world_size).
    # With micro=6 and world_size=8, the smallest valid target is 48.
    # Using target=96, micro=6 gives gradient_accumulation_steps=2 (clean and efficient).
    target_batch_size = 96   # was 36 — not divisible by (6 * 8 = 48)
    micro_batch_size  = 6
    assert target_batch_size % (micro_batch_size * ddp_world_size) == 0, (
        f"target_batch_size ({target_batch_size}) must be divisible by "
        f"micro_batch_size * world_size ({micro_batch_size} * {ddp_world_size})"
    )
    gradient_accumulation_steps = target_batch_size // (micro_batch_size * ddp_world_size)

    data_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"

    if master_process:
        print("Initializing dataset...")
    train_dataset = GPTDataset(data_path=data_path, block_size=config.block_size)

    # ==========================================
    # 3. Model Setup (no DDP yet)
    # ==========================================
    model = GPT(config)
    model.to(device)

    # ==========================================
    # 4. Optimiser & Scheduler
    # (GradScaler removed — TPU uses bfloat16 natively via xm.optimizer_step)
    # ==========================================
    start_step      = 0
    max_steps       = 610352

    learning_rate   = 5e-4
    warmup_steps    = 10000

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    warmup_scheduler  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps), eta_min=1e-5)
    scheduler         = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                     milestones=[warmup_steps])

    # ==========================================
    # 5. Checkpointing
    # ==========================================
    drive_path = "/kaggle/working/checkpoints"
    if master_process:
        os.makedirs(drive_path, exist_ok=True)

    working_checkpoint = os.path.join(drive_path, "latest_step_model.pth")
    input_checkpoint   = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
    load_path          = None

    if os.path.exists(working_checkpoint):
        load_path = working_checkpoint
    elif os.path.exists(input_checkpoint):
        load_path = input_checkpoint

    best_loss = float('inf')

    if load_path:
        if master_process:
            print(f"Loading checkpoint from: {load_path}...")
        checkpoint = torch.load(load_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"] + 1
        best_loss   = checkpoint["best_loss"]
        del checkpoint
        import gc; gc.collect()
        if master_process:
            print(f"Resumed from step {start_step}")
    else:
        if master_process:
            print("No checkpoint found. Starting from scratch!")

    # ==========================================
    # 5.5 Wrap in DDP (XLA-aware)
    # ==========================================
    model = DDP(model)
    raw_model = model.module

    # ==========================================
    # 6. Sample generation
    # ==========================================
    def generate_sample(model, device, prompt="Tell me something about AI", max_new_tokens=70):
        model.eval()
        enc    = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(prompt)
        x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        print("\n--- Model Brain Check ---")
        print(f"Prompt: '{prompt}'")

        with torch.no_grad():
            with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                for _ in range(max_new_tokens):
                    x_cond  = x[:, -config.block_size:]
                    logits  = model(x_cond)
                    logits  = logits[:, -1, :]
                    probs   = F.softmax(logits / 0.8, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                    x       = torch.cat((x, next_tok), dim=1)

        xm.mark_step()
        output_text = enc.decode(x[0].cpu().tolist())
        print(f"Output: {output_text}")
        print("----------------------------\n")
        model.train()

    # ==========================================
    # 7. Batch generator
    # ==========================================
    def create_batch_generator(dataset, micro_batch_size, ddp_rank, start_step, seed=42):
        g = torch.Generator()
        g.manual_seed(seed + ddp_rank + start_step)
        max_idx = len(dataset)
        while True:
            x_batch, y_batch = [], []
            indices = torch.randint(0, max_idx, (micro_batch_size,), generator=g).tolist()
            for idx in indices:
                x, y = dataset[idx]
                x_batch.append(x)
                y_batch.append(y)
            yield torch.stack(x_batch), torch.stack(y_batch)

    if master_process:
        print("Initializing batch generator...")

    train_iter = create_batch_generator(train_dataset, micro_batch_size, ddp_rank, start_step)

    # ==========================================
    # 8. Training loop
    # ==========================================
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    if master_process:
        print("Starting TPU training loop...")

    for step in range(start_step, max_steps):

        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        require_backward_grad_sync = (step + 1) % gradient_accumulation_steps == 0

        if not require_backward_grad_sync:
            model.require_backward_grad_sync = False
        else:
            model.require_backward_grad_sync = True

        with torch.autocast(device_type='xla', dtype=torch.bfloat16):
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
            loss = loss / gradient_accumulation_steps

        loss.backward()

        if require_backward_grad_sync:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # ==========================================
        # 9. Logging and saving (master process only)
        # BUG 3 FIX: Read loss.item() BEFORE mark_step().
        # On XLA, mark_step() flushes the compiled graph. Calling .item() after it
        # executes the *next* pending graph, so you'd log the loss from the previous
        # step. Capture the scalar value first, then flush.
        # ==========================================
        if master_process:
            real_loss = loss.item() * gradient_accumulation_steps  # <-- BEFORE mark_step()

        xm.mark_step()  # flush XLA graph AFTER extracting scalar values

        if master_process:
            if step % 10 == 0 and step > 0:
                t1 = time.time()
                dt = t1 - t0
                tokens_processed = 10 * (micro_batch_size * ddp_world_size) * config.block_size
                print(f"Step {step:5d} | Loss: {real_loss:.4f} | Speed: {(tokens_processed / dt):.2f} tok/sec")
                t0 = time.time()

            if require_backward_grad_sync:
                checkpoint = {
                    "step":                step,
                    "model_state_dict":    {k: v.cpu() for k, v in raw_model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss":           best_loss,
                }

                if step >= 400 and real_loss < best_loss and real_loss < 4.5:
                    best_loss = real_loss
                    best_path = os.path.join(drive_path, "best_model.pth")
                    torch.save(checkpoint, best_path)
                    print(f"New Best Model! Saved (Loss: {best_loss:.4f})")

                if (step + 1) % 1000 == 0:
                    interval_path = os.path.join(drive_path, "latest_step_model.pth")
                    torch.save(checkpoint, interval_path)
                    print(f"Checkpoint saved at step {step}.")
                    generate_sample(raw_model, device)

    xm.rendezvous('end')
    if master_process:
        print("Training complete!")


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    xmp.spawn(train_fn, args=(), nprocs=8)