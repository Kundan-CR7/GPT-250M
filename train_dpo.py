import os
import time
import argparse
import shutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from model import GPT, GPTConfig
from dataset import get_dpo_dataloader
from transformers import AutoTokenizer


# ============================
# LOG PROB FUNCTION
# ============================

def get_batch_logps(logits, labels, attention_mask):

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous()

    log_probs = F.log_softmax(shifted_logits, dim=-1)

    gathered = torch.gather(
        log_probs,
        dim=-1,
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)

    gathered = gathered * shifted_mask

    return gathered.sum(dim=-1)


# ============================
# DPO LOSS
# ============================

def compute_dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta
):

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits)

    margin = logits.mean().item()

    return loss.mean(), margin


# ============================
# SAFE SAVE FUNCTION
# ============================

def safe_save(checkpoint, local_path, drive_path):

    tmp_path = local_path + ".tmp"

    torch.save(checkpoint, tmp_path)

    os.replace(tmp_path, local_path)

    if drive_path is not None:

        try:
            shutil.copy(local_path, drive_path)
            print("☁️ Copied checkpoint to Drive")

        except Exception as e:
            print("⚠️ Drive copy failed:", e)


# ============================
# SAVE CHECKPOINT
# ============================

def save_checkpoint(epoch, batch_idx, model, optimizer, local_path, drive_path):

    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    safe_save(checkpoint, local_path, drive_path)

    print(f"\n💾 Model saved -> {local_path}")


# ============================
# LOAD CHECKPOINT
# ============================

def load_checkpoint(model, optimizer, path):

    if os.path.isfile(path):

        print(f"\n🔄 Loading checkpoint {path}")

        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["epoch"]

    else:

        print("\n🚀 Starting new training")

        return 0


# ============================
# MAIN
# ============================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--drive_checkpoint_dir", type=str, required=True)

    parser.add_argument("--base_weights", type=str, required=True)

    parser.add_argument("--tokenizer_name", type=str, default="gpt2")

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--grad_accum_steps", type=int, default=16)

    parser.add_argument("--save_interval", type=int, default=30)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # LOCAL CHECKPOINT DIR (FAST)
    local_ckpt_dir = "/content/checkpoints"
    os.makedirs(local_ckpt_dir, exist_ok=True)

    # DRIVE CHECKPOINT DIR (BACKUP)
    os.makedirs(args.drive_checkpoint_dir, exist_ok=True)

    local_best = os.path.join(local_ckpt_dir, "dpo_best_model.pth")
    drive_best = os.path.join(args.drive_checkpoint_dir, "dpo_best_model.pth")

    local_latest = os.path.join(local_ckpt_dir, "dpo_latest_checkpoint.pth")
    drive_latest = os.path.join(args.drive_checkpoint_dir, "dpo_latest_checkpoint.pth")

    save_interval_seconds = args.save_interval * 60

    # ============================
    # LOAD BASE MODEL
    # ============================

    base_checkpoint = torch.load(args.base_weights, map_location=device)

    if "model_args" in base_checkpoint:
        config = GPTConfig(**base_checkpoint["model_args"])
    else:
        config = GPTConfig()

    instruct_model = GPT(config).to(device)
    ref_model = GPT(config).to(device)

    state_dict = base_checkpoint.get("model_state_dict", base_checkpoint)

    instruct_model.load_state_dict(state_dict)
    ref_model.load_state_dict(state_dict)

    print("Base model loaded")

    # freeze reference model

    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad = False

    ref_model.half()

    optimizer = AdamW(instruct_model.parameters(), lr=args.lr)

    scaler = GradScaler("cuda")

    start_epoch = load_checkpoint(instruct_model, optimizer, local_latest)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_dpo_dataloader(
        jsonl_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    print("\nStarting DPO Training")

    last_save_time = time.time()

    optimizer.zero_grad()

    best_loss = 10  # prevent too many early saves

    for epoch in range(start_epoch, args.epochs):

        instruct_model.train()

        for batch_idx, batch in enumerate(train_loader):

            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)

            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            with autocast("cuda"):

                policy_chosen_logits = instruct_model(
                    chosen_ids,
                    attention_mask=chosen_mask
                )

                policy_rejected_logits = instruct_model(
                    rejected_ids,
                    attention_mask=rejected_mask
                )

                if isinstance(policy_chosen_logits, tuple):
                    policy_chosen_logits = policy_chosen_logits[0]
                    policy_rejected_logits = policy_rejected_logits[0]

                policy_chosen_logps = get_batch_logps(
                    policy_chosen_logits,
                    chosen_ids,
                    chosen_mask
                )

                policy_rejected_logps = get_batch_logps(
                    policy_rejected_logits,
                    rejected_ids,
                    rejected_mask
                )

                with torch.no_grad():

                    ref_chosen_logits = ref_model(
                        chosen_ids,
                        attention_mask=chosen_mask
                    )

                    ref_rejected_logits = ref_model(
                        rejected_ids,
                        attention_mask=rejected_mask
                    )

                    if isinstance(ref_chosen_logits, tuple):
                        ref_chosen_logits = ref_chosen_logits[0]
                        ref_rejected_logits = ref_rejected_logits[0]

                    ref_chosen_logps = get_batch_logps(
                        ref_chosen_logits,
                        chosen_ids,
                        chosen_mask
                    )

                    ref_rejected_logps = get_batch_logps(
                        ref_rejected_logits,
                        rejected_ids,
                        rejected_mask
                    )

                loss, margin = compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    args.beta
                )

                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    instruct_model.parameters(), 1.0
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

            actual_loss = loss.item() * args.grad_accum_steps

            if actual_loss < best_loss - 0.01:

                best_loss = actual_loss

                save_checkpoint(
                    epoch,
                    batch_idx,
                    instruct_model,
                    optimizer,
                    local_best,
                    drive_best
                )

                print("🏆 New best model saved")

            if batch_idx % 10 == 0:

                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Loss {actual_loss:.4f} | Margin {margin:.3f}"
                )

            current_time = time.time()

            if current_time - last_save_time >= save_interval_seconds:

                save_checkpoint(
                    epoch,
                    batch_idx,
                    instruct_model,
                    optimizer,
                    local_latest,
                    drive_latest
                )

                last_save_time = time.time()

    print("\n🎉 DPO Training Complete")


if __name__ == "__main__":
    main()