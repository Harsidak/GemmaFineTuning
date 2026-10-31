"""Quick on-device check of the whole Tunix path, before a long run:

    python -m tpu_training.probe --model-dir /kaggle/input/.../gemma-4-e4b-it-qat-q4_0-unquantized/2 \\
        --served-dir /kaggle/input/.../gemma-4-e4b-it-qat-w4a16-ct/2 [--seq 512] [--steps 2]

Stages (each reports pass/fail and seconds; a failure stops the probe with the reason):
  devices -> config -> load -> layout -> forward parity -> LoRA -> train steps -> export -> adapter contract

Exit code 0 only if every stage passed. Prints one JSON line (`PROBE_JSON:`) with the stage results, peak memory and
tokens/second, which is the number to compare against the time budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def _stage(results: list[dict[str, Any]], name: str, fn: Any) -> Any:
    t0 = time.time()
    try:
        out = fn()
        results.append({"stage": name, "ok": True, "sec": round(time.time() - t0, 1)})
        print(f"[ok]   {name:<18} {time.time() - t0:6.1f}s", flush=True)
        return out
    except BaseException as exc:
        results.append({"stage": name, "ok": False, "sec": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}"})
        print(f"[FAIL] {name:<18} {type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}", flush=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        from . import hf_config, model_io, runtime
        from .batching import batch_stats, collate
        from .loop import LoopSettings  # noqa: F401 - import check only
        from .lora_convert import peft_to_qwix
        from .peft_io import qwix_alpha, read_adapter_tensors, write_adapter_dir
        from .trainer import LoraTrainer

        devices = _stage(results, "devices", lambda: runtime.device_info())
        info["devices"] = devices
        platform = runtime.resolve_platform(args.platform)
        n_dev = devices["count"]
        parallelism = "spmd" if (platform == "tpu" and n_dev > 1) else "single"
        dtype = jnp.bfloat16 if platform == "tpu" else jnp.float32
        mesh = runtime.make_mesh(parallelism)
        info.update(platform=platform, parallelism=parallelism)

        hf = _stage(results, "config", lambda: hf_config.load_text_config(args.model_dir))
        model_cfg = hf_config.build_model_config(hf, param_dtype=dtype, dtype=dtype, remat="NONE")
        with mesh:
            model = _stage(results, "load", lambda: model_io.load_base_model(args.model_dir, model_cfg, mesh=mesh, dtype=dtype))
            _stage(results, "layout", lambda: hf_config.check_layout_against_checkpoint(model, args.model_dir))
            info["forward_parity_max_abs_diff"] = _stage(
                results, "forward parity", lambda: model_io.check_forward_parity(model, atol=5e-2 if dtype != jnp.float32 else 2e-3)
            )
            lm = _stage(results, "lora", lambda: model_io.attach_lora(
                model, rank=args.rank, alpha=qwix_alpha(args.rank, 2 * args.rank, False), seed=0))
            model_io.set_remat(lm, "BLOCK" if platform == "tpu" else "NONE")
            trainer = LoraTrainer(lm, mesh=mesh, compute_dtype=dtype, chunk_rows=4096, weight_decay=0.0, max_grad_norm=1.0)
            info["trainable_params"] = trainer.n_params

            def steps() -> dict[str, Any]:
                rng = np.random.default_rng(0)
                bsz = max(n_dev, 1) if parallelism == "spmd" else 1
                vocab = model_cfg.num_embed
                rows = [{"input_ids": rng.integers(3, min(vocab, 1000), args.seq).tolist()} for _ in range(bsz)]
                for r in rows:
                    r["labels"] = [-100] * (args.seq // 2) + r["input_ids"][args.seq // 2:]
                batch = collate(rows, batch_size=bsz, pad_id=0, pad_multiple=args.seq)
                tokens = batch_stats(batch)["tokens"]
                losses, times = [], []
                for i in range(args.steps):
                    t0 = time.time()
                    res = trainer.step([batch], lr=1e-3)
                    times.append(time.time() - t0)
                    losses.append(res["loss"])
                    print(f"       step {i}: loss={res['loss']:.4f} grad_norm={res['grad_norm']:.3f} {times[-1]:.1f}s", flush=True)
                    if not np.isfinite(res["loss"]) or not res["applied"]:
                        raise RuntimeError(f"non-finite step: {res}")
                steady = times[-1] if len(times) > 1 else None
                return {"losses": losses, "step_sec": [round(t, 2) for t in times],
                        "tokens_per_sec": round(tokens / steady, 1) if steady else None}

            info["train"] = _stage(results, "train steps", steps)
            try:
                stats = jax.local_devices()[0].memory_stats() or {}
                info["peak_hbm_gib"] = round(stats.get("peak_bytes_in_use", 0) / 1024**3, 2)
            except Exception as exc:  # noqa: BLE001 - CPU has no memory_stats
                print(f"       (no HBM stats: {type(exc).__name__})")

            def export() -> Path:
                out = Path(tempfile.mkdtemp(prefix="tunix_probe_")) / "adapter"
                peft = write_adapter_dir(out, trainer.lora_numpy(), r=args.rank, alpha=2 * args.rank)
                back = peft_to_qwix(read_adapter_tensors(out), trainer.template)  # and back: must fit the live model
                assert set(back) == set(trainer.template)
                info["adapter_tensors"] = len(peft)
                return out

            adapter_dir = _stage(results, "export", export)

        def contract() -> dict[str, Any]:
            from pipeline.multi_lora_sft.shared.adapter_contract import (
                check_adapter_contract,
            )

            rep = check_adapter_contract(adapter_dir, served_model_dir=args.served_dir, expected_rank=args.rank)
            if not rep.ok:
                raise RuntimeError("; ".join(rep.errors[:5]))
            return rep.to_dict()

        info["contract"] = _stage(results, "adapter contract", contract)
    except BaseException as exc:  # noqa: BLE001 - the failing stage already printed its reason
        info["aborted"] = f"{type(exc).__name__}: {exc}"
    ok = bool(results) and all(r["ok"] for r in results) and results[-1]["stage"] == "adapter contract"
    return {"ok": ok, "stages": results, **info}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="bf16 Hugging Face checkpoint of the model being trained")
    ap.add_argument("--served-dir", default=None, help="the w4a16 served checkpoint (enables the shape check of the adapter)")
    ap.add_argument("--platform", default="auto", choices=["auto", "tpu", "cpu"])
    ap.add_argument("--seq", type=int, default=512, help="sequence length of the synthetic batch")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    result = run(ap.parse_args(argv))
    print("PROBE_JSON:" + json.dumps(result, default=str))
    print("PROBE " + ("PASSED" if result["ok"] else "FAILED"))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
