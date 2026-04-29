"""
SFT fine-tuning for 250M GPT-2-style model on 2x T4 (Kaggle).
Run: !python sft_train.py
"""

import os, math, random
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split

import tiktoken
from datasets import load_dataset

from config import GPTConfig
from model import GPT

# ================== CONFIG ==================
CHECKPOINT_PATH = "/kaggle/input/datasets/kundan8918/gpt-250m-training-data/latest_step_model.pth"
OUTPUT_DIR      = "/kaggle/working"
BLOCK_SIZE      = 1024

EPOCHS          = 2
MICRO_BATCH     = 4
GRAD_ACCUM      = 4

LR              = 2e-5
MIN_LR          = 2e-6
WARMUP_STEPS    = 150

WEIGHT_DECAY    = 0.1
GRAD_CLIP       = 1.0

NUM_ALPACA      = 30000
NUM_OASST       = 20000

IGNORE_INDEX    = -100
EOT             = 50256


# ================== REPRO ==================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================== DATASET ==================
class ChatSFTDataset(Dataset):
    def __init__(self, block_size):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.samples = []

        print("Loading Alpaca...")
        alpaca = load_dataset("yahma/alpaca-cleaned", split=f"train[:{NUM_ALPACA}]")

        for it in alpaca:
            instr = it["instruction"].strip()
            inp   = it["input"].strip()
            out   = it["output"].strip()

            user = f"{instr}\n\n{inp}" if inp else instr
            if len(out.split()) < 3:
                continue

            self.samples.append((user, out))

        print("Loading OASST...")
        oasst = load_dataset("OpenAssistant/oasst1", split="train")

        prompts = {
            m["message_id"]: m for m in oasst
            if m["role"] == "prompter" and m["lang"] == "en"
        }

        count = 0
        for m in oasst:
            if count >= NUM_OASST:
                break

            if m["role"] == "assistant" and m["lang"] == "en" and m["parent_id"] in prompts:
                u = prompts[m["parent_id"]]["text"].strip()
                a = m["text"].strip()

                if len(a.split()) < 3:
                    continue

                self.samples.append((u, a))
                count += 1

        random.shuffle(self.samples)
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, assistant = self.samples[idx]

        prompt = f"### User:\n{user}\n\n### Assistant:\n"
        enc = self.enc

        prompt_ids = enc.encode(prompt)
        response_ids = enc.encode(assistant) + [EOT]

        full = prompt_ids + response_ids

        if len(full) > self.block_size:
            keep = self.block_size - len(prompt_ids)
            full = prompt_ids + response_ids[:keep]

        plen = len(prompt_ids)

        x = torch.tensor(full[:-1])
        y = torch.tensor(full[1:])

        y[:plen-1] = IGNORE_INDEX

        pad = (self.block_size - 1) - len(x)
        if pad > 0:
            x = torch.cat([x, torch.full((pad,), EOT)])
            y = torch.cat([y, torch.full((pad,), IGNORE_INDEX)])

        return x, y


# ================== LR ==================
def get_lr(step, total):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / (total - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# ================== DDP ==================
def setup(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


# ================== TRAIN ==================
def train(rank, world):
    ddp = world > 1
    if ddp:
        setup(rank, world)

    is_main = rank == 0
    device = f"cuda:{rank}"

    set_seed(1337 + rank)

    # ===== MODEL =====
    model = GPT(GPTConfig()).to(device)

    # ❌ REMOVED BROKEN LINE
    # model.gradient_checkpointing_enable()

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        if is_main:
            print("Loaded checkpoint")

    if ddp:
        model = DDP(model, device_ids=[rank])

    raw_model = model.module if ddp else model

    # ===== DATA =====
    dataset = ChatSFTDataset(BLOCK_SIZE)

    train_size = int(0.95 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank) if ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=MICRO_BATCH,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(val_ds, batch_size=MICRO_BATCH)

    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    optimizer.zero_grad()

    step = 0

    # ===== TRAIN LOOP =====
    for epoch in range(EPOCHS):
        if sampler:
            sampler.set_epoch(epoch)

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            is_sync = (i + 1) % GRAD_ACCUM == 0
            ctx = nullcontext() if is_sync else model.no_sync()

            with ctx:
                with torch.cuda.amp.autocast():
                    logits = model(x)

                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=IGNORE_INDEX
                    ) / GRAD_ACCUM

                scaler.scale(loss).backward()

            if is_sync:
                lr = get_lr(step, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                scaler.unscale_(optimizer)

                # ✅ FIXED HERE
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                step += 1

                if is_main and step % 20 == 0:
                    print(f"Step {step} | Loss {(loss.item()*GRAD_ACCUM):.4f} | LR {lr:.2e}")

                # ===== VALIDATION =====
                if is_main and step % 200 == 0:
                    model.eval()
                    val_loss = 0

                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(device), vy.to(device)
                            logits = model(vx)

                            l = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                vy.view(-1),
                                ignore_index=IGNORE_INDEX
                            )
                            val_loss += l.item()

                    val_loss /= len(val_loader)
                    print(f"Validation Loss: {val_loss:.4f}")
                    model.train()

                # ===== SAVE =====
                if is_main and step % 500 == 0:
                    torch.save({
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step
                    }, f"{OUTPUT_DIR}/ckpt_{step}.pth")

    if is_main:
        torch.save(raw_model.state_dict(), f"{OUTPUT_DIR}/sft_final.pth")
        print("Training complete")

    if ddp:
        cleanup()


# ================== MAIN ==================
if __name__ == "__main__":
    world = torch.cuda.device_count()

    if world == 1:
        train(0, 1)
    else:
        mp.spawn(train, args=(world,), nprocs=world)