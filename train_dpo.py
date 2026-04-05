import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# ==========================================
# 1. LOCAL IMPORTS
# ==========================================
# Ensure these match the exact class/function names in your files
from model import GPT 
from dataset import get_dpo_dataloader 
from transformers import AutoTokenizer # Used to tokenize your JSONL dataset

# ==========================================
# 2. DPO HELPER FUNCTIONS
# ==========================================
def get_batch_logps(logits, labels, attention_mask):
    """
    Extracts the log probabilities of the actual target tokens.
    """
    # Shift logits and labels for next-token prediction
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous()
    
    # Calculate log probabilities
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    
    # Gather the log probabilities of the actual labels
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out padding tokens
    gathered_log_probs = gathered_log_probs * shifted_mask
    
    # Sum log probs for each sequence in the batch
    return gathered_log_probs.sum(dim=-1)

def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    """
    Calculates the DPO loss based on the log probabilities.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()

# ==========================================
# 3. CHECKPOINTING LOGIC
# ==========================================
def save_checkpoint(epoch, batch_idx, model, optimizer, path):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"\n[💾 CHECKPOINT SAVED] Epoch: {epoch}, Batch: {batch_idx} -> {path}")

def load_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        print(f"\n[🔄 RESUMING] Loading checkpoint '{path}'...")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"[✅ LOADED] Resuming from Epoch {start_epoch}")
        return start_epoch
    else:
        print("\n[🚀 NEW RUN] No checkpoint found. Starting from scratch.")
        return 0

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run DPO Training on Custom GPT Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to your chosen/rejected JSONL dataset")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save checkpoints (e.g., Google Drive)")
    parser.add_argument("--base_weights", type=str, required=True, help="Path to pre-trained .pth weights")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="HuggingFace tokenizer ID")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=30, help="Save interval in minutes")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'dpo_latest_checkpoint.pth')
    save_interval_seconds = args.save_interval * 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nInitializing custom models on {device}...")

    # --- INITIALIZE MODELS ---
    # NOTE: If your GPT() class requires a config object (e.g., GPT(config)), pass it here.
    instruct_model = GPT().to(device) 
    ref_model = GPT().to(device) 
    
    # Load Pre-trained Base Weights
    if os.path.exists(args.base_weights):
        base_checkpoint = torch.load(args.base_weights, map_location=device)
        state_dict = base_checkpoint['model_state_dict'] if 'model_state_dict' in base_checkpoint else base_checkpoint
        
        # Load weights into both models
        instruct_model.load_state_dict(state_dict)
        ref_model.load_state_dict(state_dict)
        print(f"Loaded base weights from {args.base_weights}")
    else:
        print(f"WARNING: Base weights not found at {args.base_weights}. Initializing from scratch.")

    # Freeze the Reference Model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(instruct_model.parameters(), lr=1e-5)
    start_epoch = load_checkpoint(instruct_model, optimizer, checkpoint_path)

    # --- INITIALIZE DATALOADER ---
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    train_loader = get_dpo_dataloader(
        jsonl_path=args.data_path, 
        tokenizer=tokenizer, 
        batch_size=args.batch_size
    )

    print("\nStarting DPO Training...")
    last_save_time = time.time()
    instruct_model.train()

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader): 
            
            # 1. Unpack batch and move to device
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            optimizer.zero_grad()

            # 2. Instruct Model (Policy) Forward Pass
            # Adjust `instruct_model(...)` if your forward pass returns a tuple (e.g., logits, loss)
            policy_chosen_logits = instruct_model(chosen_ids) 
            policy_rejected_logits = instruct_model(rejected_ids)
            
            if isinstance(policy_chosen_logits, tuple):
                policy_chosen_logits = policy_chosen_logits[0]
                policy_rejected_logits = policy_rejected_logits[0]

            policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_ids, chosen_mask)
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_ids, rejected_mask)

            # 3. Reference Model Forward Pass
            with torch.no_grad():
                ref_chosen_logits = ref_model(chosen_ids)
                ref_rejected_logits = ref_model(rejected_ids)
                
                if isinstance(ref_chosen_logits, tuple):
                    ref_chosen_logits = ref_chosen_logits[0]
                    ref_rejected_logits = ref_rejected_logits[0]
                    
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_ids, chosen_mask)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_ids, rejected_mask)

            # 4. Calculate Loss & Backpropagate
            loss = compute_dpo_loss(
                policy_chosen_logps, 
                policy_rejected_logps, 
                ref_chosen_logps, 
                ref_rejected_logps, 
                args.beta
            )
            
            loss.backward()
            optimizer.step()

            # Print Progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | DPO Loss: {loss.item():.4f}")

            # 5. Timer Check
            current_time = time.time()
            if current_time - last_save_time >= save_interval_seconds:
                print(f"\n[⏱️ TIMER] {args.save_interval} minutes elapsed!")
                save_checkpoint(epoch, batch_idx, instruct_model, optimizer, checkpoint_path)
                last_save_time = time.time()
                print("Resuming training...\n")

    print("\n🎉 Custom DPO Training Complete!")

if __name__ == "__main__":
    main()