"""Step 2 of the Tunix <-> Hugging Face parity check (run in the PyTorch + transformers + peft environment).

    python tpu_training/tools/parity_check_hf.py <workdir>

Loads `<workdir>/tiny_fix` with Hugging Face, applies the adapter exported by `parity_export_tunix.py` through PEFT (exactly
what vLLM does when it serves the submission) and prints the max |logit difference| to Tunix with and without the adapter.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import Gemma4ForConditionalGeneration

S = Path(sys.argv[1]); OUT = S / "parity_out"
d = np.load(OUT / "tunix.npz")
tokens, mask = torch.tensor(d["tokens"], dtype=torch.long), torch.tensor(d["mask"], dtype=torch.long)
model = Gemma4ForConditionalGeneration.from_pretrained(str(S / "tiny_fix"), torch_dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    base = model(input_ids=tokens, attention_mask=mask).logits.numpy()
pm = PeftModel.from_pretrained(model, str(OUT / "adapter")).eval()
with torch.no_grad():
    lo = pm(input_ids=tokens, attention_mask=mask).logits.numpy()
valid = d["mask"]
def md(a, b): return float(np.abs((a - b)[valid]).max())
print("HF base   vs Tunix base :", md(base, d["logits_base"]))
print("HF + PEFT vs Tunix+LoRA :", md(lo, d["logits"]))
print("LoRA effect (Tunix)     :", md(d["logits"], d["logits_base"]), " (HF):", md(lo, base))
