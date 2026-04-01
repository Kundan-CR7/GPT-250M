import os
import requests
import numpy as np
import tiktoken

print("🚀 Starting Data Preparation Pipeline...")

# ==========================================
# 1. Download the Raw Text Data
# ==========================================
input_file_path = 'input.txt'
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

if not os.path.exists(input_file_path):
    print(f"Downloading Tiny Shakespeare dataset...")
    response = requests.get(data_url)
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Download complete!")
else:
    print("Dataset already exists locally.")

# Read the raw text into memory
with open(input_file_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

print(f"Loaded {len(text_data):,} characters of text.")

# ==========================================
# 2. Initialize the GPT-2 Tokenizer
# ==========================================
print("Initializing tiktoken (GPT-2 BPE Tokenizer)...")
# We use the exact same tokenizer OpenAI used for GPT-2 and GPT-3
enc = tiktoken.get_encoding("gpt2")

# ==========================================
# 3. Tokenize the Text (Text -> Integers)
# ==========================================
print("Encoding text into integer tokens. This might take a few seconds...")
tokens = enc.encode_ordinary(text_data)
print(f"Encoding complete! Generated {len(tokens):,} tokens.")

# ==========================================
# 4. Save to Binary (.bin)
# ==========================================
# The GPT-2 vocabulary is 50,257. 
# A 16-bit unsigned integer (uint16) can hold numbers up to 65,535.
# This means we can safely pack our tokens into uint16 to save exactly 50% of the hard drive space compared to 32-bit!
print("Saving tokens to train.bin...")
tokens_np = np.array(tokens, dtype=np.uint16)

# .tofile() writes the raw bytes directly to the hard drive. 
# This is exactly what np.memmap expects to read in our dataset.py!
tokens_np.tofile('train.bin')

print("✅ Success! train.bin has been created and is ready for the GPU.")