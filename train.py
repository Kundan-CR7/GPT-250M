import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken

# Import our custom modules
from config import GPTConfig
from dataset import GPTDataset
from model import GPT

# ==========================================
# 1. Configuration and Data Pipeline
# ==========================================
config = GPTConfig()
target_batch_size = 32
micro_batch_size = 4

print("Initializing dataset...")
train_dataset = GPTDataset(data_path="train.bin", block_size=config.block_size)

train_loader = DataLoader(
    train_dataset,
    batch_size=micro_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2
)

# ==========================================
# 2. Hardware and Model Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = GPT(config)
model.to(device)

# ==========================================
# 3. Optimization Setup
# ==========================================
gradient_accumulation_steps = target_batch_size // micro_batch_size

scaler = torch.amp.GradScaler('cuda')
learning_rate = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 4. Checkpointing Setup (Google Drive)
# ==========================================
# Make sure you run: drive.mount('/content/drive') in Colab first!
drive_path = "/content/drive/MyDrive/GPT_Project/checkpoints"
os.makedirs(drive_path, exist_ok=True)

checkpoint_path = os.path.join(drive_path, "best_model.pth")

best_loss = float('inf')
start_step = 0
max_steps = 10000  
save_interval = 500

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from Google Drive: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    start_step = checkpoint["step"] + 1
    best_loss = checkpoint["best_loss"]
    print(f"Successfully resumed! Starting from step {start_step}")
else:
    print("No checkpoint found in Drive. Starting training from scratch!")

# ==========================================
# 4.5 Evaluation Sampling Function
# ==========================================
def generate_sample(model, device, prompt="The ", max_new_tokens=30):
    # Switch to evaluation mode (turns off dropout)
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n--- 🧠 Model Brain Check ---")
    print(f"Prompt: '{prompt}'")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -config.block_size:]
            logits = model(x_cond)
            logits = logits[:, -1, :] 
            
            # Use a slightly lower temperature during training to see the most likely words
            probs = F.softmax(logits / 0.8, dim=-1) 
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
            
    output_text = enc.decode(x[0].tolist())
    print(f"Output: {output_text}")
    print("----------------------------\n")
    
    # CRITICAL: Switch back to training mode so dropout turns back on!
    model.train()

# ==========================================
# 5. The Training Loop
# ==========================================
model.train()
train_iter = iter(train_loader)
optimizer.zero_grad(set_to_none=True)
t0 = time.time()

print("Starting training loop...")
for step in range(start_step, max_steps):
    
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
        
    x, y = x.to(device), y.to(device)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        loss = loss / gradient_accumulation_steps

    scaler.scale(loss).backward()
    
    if ((step + 1) % gradient_accumulation_steps == 0):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # ==========================================
    # 6. Logging and Checkpoint Saving
    # ==========================================
    real_loss = loss.item() * gradient_accumulation_steps
    
    if step % 10 == 0 and step > 0:
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = 10 * micro_batch_size * config.block_size
        print(f"Step {step:5d} | Loss: {real_loss:.4f} | Speed: {(tokens_processed / dt):.2f} tok/sec")
        t0 = time.time() 

    if ((step + 1) % gradient_accumulation_steps == 0):
        if real_loss < best_loss:
            best_loss = real_loss
            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_loss": best_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"🌟 Saved best_model.pth to Drive! (Loss: {best_loss:.4f})")

            # Generate a sample right after saving the best model
            generate_sample(model, device, prompt="Once upon a time", max_new_tokens=30)

print("Training run completed!")