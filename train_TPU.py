import os
import time

# ============================================================
# TPU ENVIRONMENT SETUP — Must happen before any XLA imports!
# ============================================================
os.environ["PJRT_DEVICE"] = "TPU"
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import tiktoken

# Modern PyTorch XLA Imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

# Import our custom modules
from config import GPTConfig
from dataset import GPTDataset
from model import GPT


# ==========================================
# 1. The TPU Multiprocessing Wrapper
# ==========================================
def train_tpu(index):
    # Modern PJRT hardware setup
    device = torch_xla.device()
    ddp_rank = xr.global_ordinal()
    ddp_world_size = xr.world_size()
    master_process = (ddp_rank == 0)

    if master_process:
        print(f"Training on device: {device} | TPU Cores detected: {ddp_world_size}")

    # ==========================================
    # 2. Configuration and Data Pipeline
    # ==========================================
    config = GPTConfig()

    micro_batch_size = 32

    data_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"

    if master_process:
        print("Initializing dataset...")
    train_dataset = GPTDataset(data_path=data_path, block_size=config.block_size)

    # ==========================================
    # 3. Model Setup
    # ==========================================
    if master_process:
        print("Initializing model...")
    model = GPT(config)
    model.to(device)

    # ==========================================
    # 4. Optimization Setup
    # ==========================================
    start_step = 0
    max_steps = 57220
    learning_rate = 5e-4
    warmup_steps = 10000

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95)
    )
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
    # 5. Checkpointing Setup & Loading
    # ==========================================
    drive_path = "/kaggle/working/checkpoints"
    if master_process:
        os.makedirs(drive_path, exist_ok=True)

    # Sync so all ranks wait for master to create the directory
    xm.rendezvous("mkdir")

    working_checkpoint = os.path.join(drive_path, "latest_step_model.pth")
    input_checkpoint = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"

    load_path = None
    if os.path.exists(working_checkpoint):
        load_path = working_checkpoint
    elif os.path.exists(input_checkpoint):
        load_path = input_checkpoint

    best_loss = float('inf')

    if load_path:
        if master_process:
            print(f"Loading checkpoint from: {load_path}...")

        # Load to CPU first to avoid OOM on TPU
        checkpoint = torch.load(load_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        # Optionally restore optimizer & scheduler state:
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # if "scheduler_state_dict" in checkpoint:
        #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_step = checkpoint["step"] + 1
        best_loss = checkpoint.get("best_loss", float('inf'))

        del checkpoint
        import gc
        gc.collect()

        if master_process:
            print(f"Successfully resumed! Starting from step {start_step}")
    else:
        if master_process:
            print("No checkpoint found. Starting training from scratch!")

    # ==========================================
    # 6. Evaluation / Sampling Function
    # ==========================================
    def generate_sample(model, device, master_process,
                        prompt="Tell me something about AI",
                        max_new_tokens=70):
        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        if master_process:
            print("\n--- Model Brain Check ---")
            print(f"Prompt: '{prompt}'")

        with torch.no_grad():
            for _ in range(max_new_tokens):
                seq_len = x.size(1)
                start_idx = max(0, seq_len - config.block_size)
                x_cond = x[:, start_idx:]
                logits = model(x_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
                # Step the graph after each token to avoid giant graphs
                xm.mark_step()

        if master_process:
            output_text = enc.decode(x[0].tolist())
            print(f"Output: {output_text}")
            print("----------------------------\n")

        model.train()

    # ==========================================
    # 7. Batch Generator
    # ==========================================
    def create_batch_generator(dataset, micro_batch_size, ddp_rank,
                               start_step, seed=42):
        g = torch.Generator()
        g.manual_seed(seed + ddp_rank + start_step)
        max_idx = len(dataset)

        while True:
            x_batch, y_batch = [], []
            indices = torch.randint(
                0, max_idx, (micro_batch_size,), generator=g
            ).tolist()
            for idx in indices:
                x, y = dataset[idx]
                x_batch.append(x)
                y_batch.append(y)
            yield torch.stack(x_batch), torch.stack(y_batch)

    # ==========================================
    # 8. Distributed Training Loop
    # ==========================================
    model.train()
    train_iter = create_batch_generator(
        train_dataset, micro_batch_size, ddp_rank, start_step
    )
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    if master_process:
        print("Starting TPU distributed training loop...")

    for step in range(start_step, max_steps):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # TPU bfloat16 forward pass
        with torch.autocast('xla', dtype=torch.bfloat16):
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # XLA optimizer step — fuses allreduce + weight update
        xm.optimizer_step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Tell the TPU to execute the compiled graph now.
        # Prevents multi-step "mega-graph" buildup.
        xm.mark_step()

        # ==========================================
        # 9. Logging
        # ==========================================
        if step % 10 == 0:
            # .item() implicitly calls mark_step / syncs the value
            real_loss = loss.detach().item()

            if master_process:
                if step > 0:
                    t1 = time.time()
                    dt = t1 - t0
                    tokens_processed = (
                        10 * (micro_batch_size * ddp_world_size) * config.block_size
                    )
                    tok_per_sec = tokens_processed / dt
                    print(
                        f"Step {step:5d} | "
                        f"Loss: {real_loss:.4f} | "
                        f"Speed: {tok_per_sec:.2f} tok/sec"
                    )
                t0 = time.time()

        # ==========================================
        # 10. Interval Checkpointing
        # ==========================================
        if (step + 1) % 500 == 0:
            # Sync all ranks before saving
            xm.rendezvous("pre_save_sync")

            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            }

            interval_path = os.path.join(drive_path, f"model_step_{step + 1}.pth")
            # xm.save handles cross-rank coordination automatically
            xm.save(checkpoint, interval_path)

            # Also overwrite the "latest" checkpoint for easy resuming
            xm.save(checkpoint, working_checkpoint)

            if master_process:
                print(f"Saved checkpoint to {interval_path}")

            generate_sample(model, device, master_process)


# ==========================================
# TPU Launch — use 'spawn', NOT 'fork'
# 'fork' causes the TPU runtime metrics crash:
#   "Check failed: reporting_closure_ == nullptr"
# because forked children inherit already-initialized
# TPU state from the parent process.
# ==========================================
if __name__ == '__main__':
    xmp.spawn(train_tpu, args=(), nprocs=None, start_method='spawn')