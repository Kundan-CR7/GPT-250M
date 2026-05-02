import json
from datasets import load_dataset

# We want exactly the numbers from your training config
NUM_ALPACA = 30000
NUM_OASST = 20000

training_data = []

# ==========================================
# 1. Grab EXACTLY 30,000 Alpaca Rows
# ==========================================
print(f"Slicing exactly {NUM_ALPACA} rows from Alpaca...")
alpaca = load_dataset("yahma/alpaca-cleaned", split=f"train[:{NUM_ALPACA}]")

for it in alpaca:
    instr = it["instruction"].strip()
    inp   = it["input"].strip()
    out   = it["output"].strip()

    # Format exactly how your model sees it
    user = f"{instr}\n\n{inp}" if inp else instr
    
    if len(out.split()) >= 3:
        training_data.append({
            "source": "alpaca",
            "user_prompt": user,
            "assistant_response": out
        })

# ==========================================
# 2. Grab EXACTLY 20,000 OpenAssistant Rows
# ==========================================
print(f"Extracting exactly {NUM_OASST} rows from OASST...")
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

        if len(a.split()) >= 3:
            training_data.append({
                "source": "openassistant",
                "user_prompt": u,
                "assistant_response": a
            })
            count += 1

# ==========================================
# 3. Save to a perfectly formatted JSON file
# ==========================================
output_filename = "exact_50k_training_mix.json"
print(f"Saving a total of {len(training_data)} rows to {output_filename}...")

with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=4, ensure_ascii=False)

print("✅ Done! You now have the exact 50,000 rows your model will be trained on.")