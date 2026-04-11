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

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from config import GPTConfig
from dataset import GPTDataset
from model import GPT


def train_tpu(index):

    # ==========================================
    # 1. Hardware Setup
    # ==========================================
    device    = torch_xla.device()
    ddp_rank  = xr.global_ordinal()
    ddp_world = xr.world_size()
    is_master = (ddp_rank == 0)

    if is_master:
        print(f"Device: {device} | Cores: {ddp_world}")

    # ==========================================
    # 2. Hyperparameters
    # ==========================================
    config           = GPTConfig()
    micro_batch_size = 4        # 4 per core x 8 cores = 32 effective batch
                                # tokens/step = 4 x 8 x 1024 = 32,768
                                # steps for 5B tokens = 5B / 32,768 = 152,587

    max_steps    = 152_587
    warmup_steps = 7_629        # 5% of max_steps
    max_lr       = 3e-4
    min_lr       = 3e-5
    weight_decay = 0.1
    grad_clip    = 1.0

    # ==========================================
    # 3. Data
    # ==========================================
    data_path = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/train.bin"
    if is_master:
        print("Loading dataset...")
    train_dataset = GPTDataset(data_path=data_path, block_size=config.block_size)

    # ==========================================
    # 4. Model
    # ==========================================
    if is_master:
        print("Building model...")
    model = GPT(config)
    model.to(device)

    if is_master:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count / 1e6:.1f}M")

    # ==========================================
    # 5. Optimizer + SequentialLR
    # ==========================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        eps=1e-8,
    )
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(max_steps - warmup_steps),
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ==========================================
    # 6. Checkpointing paths
    # ==========================================
    ckpt_dir = "/kaggle/working/checkpoints"
    if is_master:
        os.makedirs(ckpt_dir, exist_ok=True)
    xm.rendezvous("mkdir")

    latest_ckpt = os.path.join(ckpt_dir, "latest_step_model.pth")

    # ==========================================
    # 7. Resume logic
    # ==========================================
    start_step = 0

    if os.path.exists(latest_ckpt):
        if is_master:
            print(f"Resuming from {latest_ckpt} ...")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_step = ckpt.get("step", 0) + 1
        del ckpt

        import gc
        gc.collect()

        if is_master:
            print(f"Resumed successfully at step {start_step}")
    else:
        if is_master:
            print("No checkpoint found — training from scratch.")

    # ==========================================
    # 8. Batch generator
    # ==========================================
    def batch_generator(dataset, batch_size, rank, start, seed=42):
        g = torch.Generator()
        g.manual_seed(seed + rank + start)
        n = len(dataset)
        while True:
            xs, ys = [], []
            for idx in torch.randint(0, n, (batch_size,), generator=g).tolist():
                x, y = dataset[idx]
                xs.append(x)
                ys.append(y)
            yield torch.stack(xs), torch.stack(ys)

    # ==========================================
    # 9. Text generation for sanity-checking
    # ==========================================
    def generate_sample(prompt="Tell me something about AI", max_new_tokens=30):
        model.eval()
        enc    = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(prompt)
        x      = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        if is_master:
            print("\n--- Sample ---")
            print(f"Prompt: '{prompt}'")

        with torch.no_grad():
            for _ in range(max_new_tokens):
                sl   = x.size(1)
                # safe slice — avoids TPU out-of-bounds on short sequences
                cond     = x[:, max(0, sl - config.block_size):]
                logits   = model(cond)
                logits   = logits[:, -1, :]
                probs    = F.softmax(logits / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                x        = torch.cat((x, next_tok), dim=1)
                xm.mark_step()

        if is_master:
            print(f"Output: {enc.decode(x[0].tolist())}")
            print("--------------\n")

        model.train()

    # ==========================================
    # 10. Training loop
    # ==========================================
    model.train()
    data_iter = batch_generator(train_dataset, micro_batch_size, ddp_rank, start_step)
    optimizer.zero_grad()

    loss_val = float("inf")

    t0 = time.time()

    if is_master:
        tps = micro_batch_size * ddp_world * config.block_size
        print(f"Tokens/step : {tps:,}")
        print(f"Total steps : {max_steps:,}")
        print(f"Total tokens: {tps * max_steps / 1e9:.2f}B")
        print(f"Starting at step {start_step}")

    for step in range(start_step, max_steps):

        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        with torch.autocast("xla", dtype=torch.bfloat16):
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        xm.optimizer_step(optimizer)
        
        # FIX: standard zero_grad to keep TPU memory pointers static
        optimizer.zero_grad()
        scheduler.step()

        # mark_step AFTER scheduler.step()
        xm.mark_step()

        # -- logging every 10 steps --
        if step % 10 == 0:
            loss_val = loss.detach().item()
            cur_lr   = scheduler.get_last_lr()[0]

            if is_master:
                if step > start_step:
                    dt      = time.time() - t0
                    tok_sec = 10 * micro_batch_size * ddp_world * config.block_size / dt
                    print(
                        f"Step {step:7d}/{max_steps} | "
                        f"Loss: {loss_val:.4f} | "
                        f"LR: {cur_lr:.2e} | "
                        f"Speed: {tok_sec:,.0f} tok/sec"
                    )
                t0 = time.time()

        # -- checkpoint every 1000 steps --
        if (step + 1) % 1000 == 0:
            xm.rendezvous("pre_save")

            if (step % 10) != 0:
                loss_val = loss.detach().item()

            ckpt = {
                "step":                 step,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            
            # This safely overwrites the exact same file every time
            xm.save(ckpt, latest_ckpt)

            if is_master:
                print(f"💾 Overwrote latest_step_model.pth (Current Loss: {loss_val:.4f})")

            generate_sample()

    if is_master:
        print("Training complete!")


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    xmp.spawn(train_tpu, args=(), nprocs=None, start_method="spawn")