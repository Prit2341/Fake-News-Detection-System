"""
VRAM usage tester for GTX 1650 (4GB).
Tests Sentence Transformer + XLM-RoBERTa at different batch sizes and sequence lengths.
Recommends safe settings before full training.
"""

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

def vram_used():
    return torch.cuda.memory_allocated() / 1024**2

def vram_peak():
    return torch.cuda.max_memory_allocated() / 1024**2

def vram_total():
    return torch.cuda.get_device_properties(0).total_memory / 1024**2

def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def print_vram(label):
    used  = vram_used()
    peak  = vram_peak()
    safe  = vram_total() * 0.85
    flag  = " ⚠ EXCEEDS PHYSICAL VRAM" if peak > vram_total() else " ⚠ NEAR LIMIT" if peak > safe else ""
    print(f"  {label:<45} used={used:.0f}MB  peak={peak:.0f}MB{flag}")

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─── Info ─────────────────────────────────────────────────────
total = vram_total()
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {total:.0f} MB ({total/1024:.1f} GB)")
print(f"Safe limit: {total * 0.85:.0f} MB (85% of total)")

# ─── Sentence Transformer ─────────────────────────────────────
separator("Sentence Transformer: paraphrase-multilingual-MiniLM-L12-v2")

clear()
print("  Loading model...")
smodel = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
print_vram("Model loaded")

dummy_texts = ["This is a test news article about politics and elections."] * 1

for batch_size in [32, 64, 128, 256, 512]:
    clear()
    torch.cuda.reset_peak_memory_stats()
    texts = dummy_texts * batch_size
    try:
        _ = smodel.encode(texts, batch_size=batch_size, show_progress_bar=False)
        status = "OK"
    except RuntimeError as e:
        if "out of memory" in str(e):
            status = "OOM"
        else:
            status = f"ERR: {e}"
    label = f"Batch size {batch_size}"
    if status == "OK":
        print_vram(f"{label:<20} → {status}")
    else:
        print(f"  {label:<20} → {status}")

del smodel
clear()

# ─── XLM-RoBERTa Inference ────────────────────────────────────
separator("XLM-RoBERTa Inference (predict only)")

print("  Loading model...")
clear()
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model     = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2).to(device)
model.eval()
print_vram("Model loaded")

dummy = ["Breaking news about government policy and elections." * 5]

for seq_len in [128, 256, 512]:
    for batch_size in [8, 16, 32]:
        clear()
        texts = dummy * batch_size
        enc   = tokenizer(texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    _ = model(input_ids=enc["input_ids"].to(device),
                              attention_mask=enc["attention_mask"].to(device))
            print_vram(f"seq={seq_len}  batch={batch_size:<4} → OK")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  seq={seq_len}  batch={batch_size:<4} → OOM")
            clear()

# ─── XLM-RoBERTa Training (forward + backward) ────────────────
separator("XLM-RoBERTa Training (forward + backward pass)")

model.train()
scaler = torch.cuda.amp.GradScaler()

for seq_len in [128, 256]:
    for batch_size in [4, 8, 16]:
        clear()
        texts  = dummy * batch_size
        enc    = tokenizer(texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
        labels = torch.zeros(batch_size, dtype=torch.long).to(device)
        try:
            with torch.amp.autocast("cuda"):
                out  = model(input_ids=enc["input_ids"].to(device),
                             attention_mask=enc["attention_mask"].to(device),
                             labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(torch.optim.AdamW(model.parameters(), lr=2e-5))
            scaler.update()
            print_vram(f"seq={seq_len}  batch={batch_size:<4} → OK  (TRAIN)")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  seq={seq_len}  batch={batch_size:<4} → OOM (TRAIN)")
            clear()

del model
clear()

# ─── Summary ──────────────────────────────────────────────────
separator("Recommendation Summary")
print(f"""
  GPU Total VRAM : {total:.0f} MB
  Safe budget    : {total * 0.85:.0f} MB

  Check above results and pick the highest batch size
  that shows OK without going over safe budget.

  Typical safe settings for GTX 1650 (4GB):
    Sentence Transformer  → batch_size = 128-256
    XLM-RoBERTa inference → seq=256, batch=16
    XLM-RoBERTa training  → seq=256, batch=8  (fp16)
""")
