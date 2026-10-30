"""Backend-agnostic training practices shared by every trainer (TRL/PyTorch on GPU, Tunix/JAX on TPU).

Nothing here imports torch, jax, transformers or tunix, so the same code (and the same tests) covers every backend:
atomic IO, checkpoint publishing and verification, NaN / divergence guards, best-checkpoint selection, resume
compatibility, the task-level progress ledger, tracking (W&B) policy, Optuna tuning, secret scanning, the adapter
contract check, and the cross-session checkpoint bundle.
"""
