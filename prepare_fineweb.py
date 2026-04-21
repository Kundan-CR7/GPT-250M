"""
One-time tokenization of HuggingFaceFW/fineweb-edu into a single train.bin.
Saves as uint16 (GPT-2 vocab fits in 16 bits). Streams from HF, no RAM blowup.
After it finishes: click 'Save Version' → 'Save & Run All' → this creates
a Kaggle Dataset containing /kaggle/working/train.bin that you'll use in Notebook B.
"""
import os, numpy as np, tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_PATH     = "/kaggle/working/train.bin"
TOTAL_TOKENS = 10_000_000_000      # 10B tokens → ~20GB
SUBSET       = "sample-100BT"

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

print(f"Streaming {SUBSET}; target {TOTAL_TOKENS:,} tokens → {OUT_PATH}")
ds = load_dataset("HuggingFaceFW/fineweb-edu", name=SUBSET,
                  split="train", streaming=True)

# Pre-allocate a 20GB memory-mapped file and fill it in order
arr = np.memmap(OUT_PATH, dtype=np.uint16, mode="w+", shape=(TOTAL_TOKENS,))
idx = 0
pbar = tqdm(total=TOTAL_TOKENS, unit="tok", unit_scale=True)

buf = []
BUF_FLUSH = 1_000_000   # flush every ~1M tokens to reduce memmap write overhead

for doc in ds:
    if idx >= TOTAL_TOKENS: break
    toks = [EOT] + enc.encode_ordinary(doc["text"])
    buf.extend(toks)
    if len(buf) >= BUF_FLUSH:
        n = min(len(buf), TOTAL_TOKENS - idx)
        arr[idx : idx + n] = np.asarray(buf[:n], dtype=np.uint16)
        idx += n
        pbar.update(n)
        buf = buf[n:]
        if idx >= TOTAL_TOKENS: break

# Flush any remainder
if buf and idx < TOTAL_TOKENS:
    n = min(len(buf), TOTAL_TOKENS - idx)
    arr[idx : idx + n] = np.asarray(buf[:n], dtype=np.uint16)
    idx += n
    pbar.update(n)

arr.flush()
del arr
pbar.close()
print(f"Done. Wrote {idx:,} tokens to {OUT_PATH}")
print(f"File size: {os.path.getsize(OUT_PATH) / 1e9:.2f} GB")