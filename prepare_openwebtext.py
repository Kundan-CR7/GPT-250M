import os
import numpy as np
import tiktoken
from datasets import load_dataset

print("🚀 Connecting to OpenWebText...")

# ==========================================
# 1. Stream the Dataset
# ==========================================
# streaming=True means we download documents one by one over the network, 
# preventing Colab's RAM and Hard Drive from exploding.
dataset = load_dataset("openwebtext", split="train", streaming=True)

# ==========================================
# 2. Setup the Tokenizer
# ==========================================
enc = tiktoken.get_encoding("gpt2")

# We must tell GPT-2 when one article ends and a completely unrelated one begins.
# We do this by inserting a special <|endoftext|> token between documents.
EOT_TOKEN = enc._special_tokens['<|endoftext|>']

target_tokens = 100_000_000  # Your target: 100 Million tokens
total_tokens = 0
all_tokens = []

print(f"Downloading and tokenizing up to {target_tokens:,} tokens...")
print("This will take a few minutes as we stream from the Hugging Face servers...")

# ==========================================
# 3. Process on the Fly
# ==========================================
for doc in dataset:
    # Encode the raw text
    tokens = enc.encode_ordinary(doc['text'])
    
    # Append the End Of Text token
    tokens.append(EOT_TOKEN)
    
    # Add to our master list
    all_tokens.extend(tokens)
    total_tokens += len(tokens)
    
    # Stop streaming once we hit our goal
    if total_tokens >= target_tokens:
        break

# Trim the list down to exactly 100M in case the last article pushed us over
all_tokens = all_tokens[:target_tokens]

print(f"Collected exactly {len(all_tokens):,} tokens.")

# ==========================================
# 4. Save to Disk
# ==========================================
print("Saving to train.bin...")

# Convert to 16-bit integers to save 50% RAM/Disk space
tokens_np = np.array(all_tokens, dtype=np.uint16)
tokens_np.tofile('train.bin')

print("✅ Success! Your OpenWebText train.bin is ready for the GPU.")