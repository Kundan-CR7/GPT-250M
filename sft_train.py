"""
SFT fine-tuning for 250M GPT-2-style model on 2x T4 (Kaggle).
Run: !python sft_train.py
"""
import os, math, random
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
import tiktoken
from datasets import load_dataset

from config import GPTConfig
from model import GPT

# ================== CONFIG ==================
CHECKPOINT_PATH = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
OUTPUT_DIR      = "/kaggle/working"
BLOCK_SIZE      = 1024

EPOCHS          = 2
MICRO_BATCH     = 4        # per GPU
GRAD_ACCUM      = 4        # effective = 4 * 4 * 2 = 32
LR              = 1e-5
MIN_LR          = 1e-6
WARMUP_STEPS    = 150
WEIGHT_DECAY    = 0.1
GRAD_CLIP       = 1.0

NUM_ALPACA      = 30000
NUM_OASST       = 20000
REPLAY_PATH     = None     # set to a .txt of pretraining text to prevent forgetting
REPLAY_RATIO    = 0.10

IGNORE_INDEX    = -100
EOT             = 50256    # GPT-2 <|endoftext|>

# ================== DATASETS ==================
class ChatSFTDataset(Dataset):
    """Alpaca-style format — uses only tokens in vanilla GPT-2 vocab."""
    def __init__(self, block_size):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.samples = []

        print("Loading Alpaca...")
        alpaca = load_dataset("yahma/alpaca-cleaned", split=f"train[:{NUM_ALPACA}]")
        for it in alpaca:
            instr, inp, out = it["instruction"].strip(), it["input"].strip(), it["output"].strip()
            user = f"{instr}\n\n{inp}" if inp else instr
            if user and out:
                self.samples.append((user, out))

        print("Loading OASST1...")
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        prompts = {m["message_id"]: m for m in oasst
                   if m["role"] == "prompter" and m["lang"] == "en"}
        n = 0
        for m in oasst:
            if n >= NUM_OASST: break
            if m["role"] == "assistant" and m["lang"] == "en" and m["parent_id"] in prompts:
                u = prompts[m["parent_id"]]["text"].strip()
                a = m["text"].strip()
                if u and a:
                    self.samples.append((u, a))
                    n += 1

        random.shuffle(self.samples)
        print(f"Total SFT samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, assistant = self.samples[idx]
        prompt_str   = f"### User:\n{user}\n\n### Assistant:\n"
        response_str = assistant

        prompt_ids   = self.enc.encode(prompt_str)
        response_ids = self.enc.encode(response_str) + [EOT]

        full = prompt_ids + response_ids
        if len(full) > self.block_size:
            # Truncate response end, keep prompt intact
            keep = self.block_size - len(prompt_ids)
            if keep <= 1:
                # Prompt itself too long — hard-truncate prompt from the left
                prompt_ids = prompt_ids[-(self.block_size // 2):]
                keep = self.block_size - len(prompt_ids)
            full = prompt_ids + response_ids[:keep]

        plen = len(prompt_ids)
        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:],  dtype=torch.long)
        y[: plen - 1] = IGNORE_INDEX  # loss only on assistant tokens

        pad = (self.block_size - 1) - len(x)
        if pad > 0:
            x = torch.cat([x, torch.full((pad,), EOT, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)])
        return x, y


class ReplayDataset(Dataset):
    """Pretraining text replay — prevents catastrophic forgetting."""
    def __init__(self, text_path, block_size, num_samples):
        enc = tiktoken.get_encoding("gpt2")
        with open(text_path, "r") as f:
            self.tokens = torch.tensor(enc.encode(f.read()), dtype=torch.long)
        self.block_size = block_size
        self.num_samples = num_samples

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        i = random.randint(0, len(self.tokens) - self.block_size - 1)
        x = self.tokens[i : i + self.block_size - 1].clone()
        y = self.tokens[i + 1 : i + self.block_size].clone()
        return x, y  # no mask — train on everything

# ================== LR / OPTIM ==================
def get_lr(step, total):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(prog, 1.0)))

def build_optimizer(model):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad: continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95), eps=1e-8,
    )

# ================== TRAIN ==================
def setup(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

def cleanup(): dist.destroy_process_group()

def train(rank, world):
    ddp = world > 1
    if ddp: setup(rank, world)
    is_main = rank == 0
    device = f"cuda:{rank}"
    torch.manual_seed(1337 + rank)

    # Model
    model = GPT(GPTConfig()).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        if is_main: print(f"Loaded base: {CHECKPOINT_PATH}")
    if ddp:
        model = DDP(model, device_ids=[rank])
    raw_model = model.module if ddp else model

    # Data
    sft = ChatSFTDataset(BLOCK_SIZE)
    parts = [sft]
    if REPLAY_PATH and os.path.exists(REPLAY_PATH):
        n = int(len(sft) * REPLAY_RATIO / (1 - REPLAY_RATIO))
        parts.append(ReplayDataset(REPLAY_PATH, BLOCK_SIZE, n))
        if is_main: print(f"Added {n} replay samples")
    train_ds = ConcatDataset(parts) if len(parts) > 1 else sft

    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(train_ds, batch_size=MICRO_BATCH, sampler=sampler,
                        shuffle=(sampler is None), num_workers=2, pin_memory=True, drop_last=True)

    total_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
    optimizer = build_optimizer(raw_model)
    scaler    = torch.cuda.amp.GradScaler()  # <-- THE critical missing piece

    if is_main:
        tp = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"Params: {tp/1e6:.1f}M | Total steps: {total_steps} | Warmup: {WARMUP_STEPS}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    gstep = 0

    for epoch in range(EPOCHS):
        if sampler: sampler.set_epoch(epoch)
        for mstep, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_sync = (mstep + 1) % GRAD_ACCUM == 0
            sync_ctx = nullcontext() if (is_sync or not ddp) else model.no_sync()

            with sync_ctx:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1), ignore_index=IGNORE_INDEX,
                    ) / GRAD_ACCUM
                scaler.scale(loss).backward()

            if is_sync:
                lr = get_lr(gstep, total_steps)
                for g in optimizer.param_groups: g["lr"] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                gstep += 1

                if is_main and gstep % 20 == 0:
                    print(f"ep{epoch} step {gstep}/{total_steps} "
                          f"loss {loss.item()*GRAD_ACCUM:.4f} lr {lr:.2e}")
                if is_main and gstep % 500 == 0:
                    torch.save(raw_model.state_dict(), f"{OUTPUT_DIR}/sft_step_{gstep}.pth")

    if is_main:
        torch.save(raw_model.state_dict(), f"{OUTPUT_DIR}/sft_final.pth")
        print("Done.")
    if ddp: cleanup()


if __name__ == "__main__":
    world = torch.cuda.device_count()
    assert world >= 1, "No GPU"
    if world == 1:
        train(0, 1)
    else:
        mp.spawn(train, args=(world,), nprocs=world, join=True)
