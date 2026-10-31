"""Step 1 of the Tunix <-> Hugging Face parity check (run in the JAX environment).

    python tpu_training/tools/parity_export_tunix.py <workdir>      # <workdir>/tiny_fix = a tiny Gemma 4 checkpoint

Attaches a random-weight LoRA to the checkpoint in Tunix, writes it as a PEFT adapter (the same writer the trainer uses)
and saves the Tunix logits for fixed tokens. Step 2 (`parity_check_hf.py`, PyTorch + PEFT environment) loads that adapter
into the Hugging Face model and compares logits; they must agree to float32 rounding (1e-5). Verified at 1.7e-6.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tunix.sft import utils as su

from tpu_training import hf_config, model_io, peft_io, runtime

S = Path(sys.argv[1]); FIX = S / "tiny_fix"; OUT = S / "parity_out"; OUT.mkdir(exist_ok=True)
hf = hf_config.load_text_config(FIX)
cfg = hf_config.build_model_config(hf, param_dtype=jnp.float32, dtype=jnp.float32, remat="NONE")
model = model_io.load_base_model(FIX, cfg, mesh=runtime.make_mesh("single"), dtype=jnp.float32)
R, ALPHA = 4, 8.0
lm = model_io.attach_lora(model, rank=R, alpha=peft_io.qwix_alpha(R, ALPHA, False), seed=0)
gd, lora, rest = model_io.split_model(lm)
rng = np.random.default_rng(7)
arrs = {k: (rng.standard_normal(v.shape) * 0.08).astype(np.float32) for k, v in model_io.lora_to_numpy(lora).items()}
lora = model_io.numpy_to_lora(lora, arrs)
peft_io.write_adapter_dir(OUT / "adapter", arrs, r=R, alpha=ALPHA, base_model="tiny")
tokens = np.array([[1,5,9,12,3,7,2,44,81,6,3,90], [1,17,30,21,3,7,2,0,0,0,0,0]], dtype=np.int32)
mask = tokens != 0; mask[:, 0] = True
m = nnx.merge(gd, lora, rest)
attn = su.make_causal_attn_mask(jnp.asarray(mask)); pos = su.build_positions_from_mask(jnp.asarray(mask))
logits = np.asarray(model_io.logits_from_hidden(m, model_io.hidden_states(m, jnp.asarray(tokens), pos, attn)))
base = nnx.merge(gd, jax.tree.map(jnp.zeros_like, lora), rest)
logits0 = np.asarray(model_io.logits_from_hidden(base, model_io.hidden_states(base, jnp.asarray(tokens), pos, attn)))
np.savez(OUT / "tunix.npz", tokens=tokens, mask=mask, logits=logits, logits_base=logits0)
print("tunix logits", logits.shape, "lora effect max|diff|", float(np.abs(logits - logits0).max()))
