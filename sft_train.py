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

# Normal LR for full Fine-Tuning
LR              = 2e-5  
MIN_LR          = 5e-7
WARMUP_STEPS    = 150   

WEIGHT_DECAY    = 0.1
GRAD_CLIP       = 1.0

# Using a safe slice of the massive NVIDIA dataset
NUM_NEMOTRON    = 30000 

IGNORE_INDEX    = -100
EOT             = 50256


# ================== REPRO ==================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================== DATASET ==================
# ================== DATASET ==================
class ChatSFTDataset(Dataset):
    def __init__(self, block_size):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.samples = []

        print("Loading NVIDIA Nemotron Data via Streaming (Zero Disk Space)...")
        try:
            # 🚨 CHANGED: Added streaming=True so it doesn't download 100GB of files!
            nvidia_data = load_dataset(
                "nvidia/Nemotron-Cascade-2-SFT-Data", 
                "chat", 
                split="train",
                streaming=True 
            )
            
            loaded_count = 0
            
            for row in nvidia_data:
                # 🚨 CHANGED: We manually stop the stream once we hit our target
                if loaded_count >= NUM_NEMOTRON:
                    break
                    
                messages = row["messages"]
                
                user_text = ""
                assistant_text = ""
                
                for msg in messages:
                    if msg["role"] == "user":
                        user_text = msg["content"].strip()
                    elif msg["role"] == "assistant":
                        assistant_text = msg["content"].strip()
                        
                        if user_text and assistant_text and len(assistant_text.split()) >= 3:
                            self.samples.append((user_text, assistant_text))
                            loaded_count += 1 # Count the successful pairs!
                            
                            user_text = ""
                            assistant_text = ""
                            
            print(f"✅ Successfully streamed {len(self.samples)} turns from NVIDIA!")
        except Exception as e:
            print(f"⚠️ Failed to load NVIDIA data: {e}")

        random.shuffle(self.samples)
        print(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, assistant = self.samples[idx]

        prompt = f"### User:\n{user}\n\n### Assistant:\n"
        enc = self.enc

        prompt_ids = enc.encode(prompt)
        response_ids = enc.encode(assistant) + [EOT]

        full = prompt_ids + response_ids
        full = full[:self.block_size]

        if len(full) < 2:
            full = full + [EOT]

        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:], dtype=torch.long)

        plen = min(len(prompt_ids), len(y))
        y[:plen] = IGNORE_INDEX

        pad_len = self.block_size - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), EOT, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])

        x = x[:self.block_size]
        y = y[:self.block_size]

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


# ================== LOAD CHECKPOINT ==================
def load_checkpoint(model, path, is_main):
    if not os.path.exists(path):
        if is_main:
            raise FileNotFoundError(f"🚨 CRITICAL ERROR: Could not find checkpoint at {path}! Stopping training so we don't start from scratch.")
        return

    try:
        ckpt = torch.load(path, map_location="cpu")

        if "model" in ckpt:
            state = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)

        if is_main:
            print("✅ Checkpoint loaded")
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")

    except Exception as e:
        if is_main:
            print("⚠️ Checkpoint incompatible, training from scratch")
            print("Error:", e)


# ================== TRAIN ==================
def train(rank, world):
    ddp = world > 1
    if ddp:
        setup(rank, world)

    is_main = rank == 0
    device = f"cuda:{rank}"

    set_seed(1337 + rank)

    model = GPT(GPTConfig()).to(device)

    load_checkpoint(model, CHECKPOINT_PATH, is_main)

    if ddp:
        model = DDP(model, device_ids=[rank])

    raw_model = model.module if ddp else model

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
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                step += 1

                if is_main and step % 20 == 0:
                    print(f"Step {step} | Loss {(loss.item()*GRAD_ACCUM):.4f} | LR {lr:.2e}")

    if is_main:
        torch.save(raw_model.state_dict(), f"{OUTPUT_DIR}/sft_final.pth")
        print("✅ Training complete")

    if ddp:
        cleanup()


# ================== MAIN ==================
if __name__ == "__main__":
    world = torch.cuda.device_count()

    if world == 1:
        train(0, 1)
    else:
        mp.spawn(train, args=(world,), nprocs=world)