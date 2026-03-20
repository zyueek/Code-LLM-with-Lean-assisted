#!/usr/bin/env python3
"""
SFT warm-start for DeepSeek Prover as an ITSSM *critic* (Stage-2 combined tags).

What it trains
--------------
Given:
  - problem statement
  - candidate draft code

Train the prover to output ONLY:
  <subgoal>one per line</subgoal>
  <gap_analysis>bullets</gap_analysis>
  <checklist>bullets</checklist>

Why SFT first
-------------
RL/GRPO is high variance; starting from a model already competent at the *exact*
format and role used in evaluation significantly improves stability and prevents
reward hacking of the downstream coder.

Data
----
Uses the ITSSM JSONL files produced by `generate_itssm_rl_dataset_livecodebench.py`,
typically named `itssm_lcb_v*.jsonl`.

Output
------
Writes a PEFT adapter directory (LoRA/QLoRA) to --output-dir, plus a `base_model.json`
hint used by downstream evaluation scripts.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from rl_training_prover_process import build_prover_stage2_prompt, _as_bullets, extract_itssm_targets


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def _primary(self):
        for s in self._streams:
            if s is not None:
                return s
        raise AttributeError("No underlying stream")

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def fileno(self):
        return self._primary().fileno()

    def isatty(self):
        try:
            return bool(self._primary().isatty())
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._primary(), name)


def _setup_logging(log_path: Optional[Path]) -> Optional[object]:
    """
    Configure python logging + tee stdout/stderr to a file (rank0 only).
    Returns the opened file handle that must stay alive for the process lifetime.
    """
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except Exception:
        rank = 0
    if rank != 0 or log_path is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")

    sys.stdout = _Tee(sys.stdout, f)  # type: ignore
    sys.stderr = _Tee(sys.stderr, f)  # type: ignore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(str(log_path), encoding="utf-8")],
        force=True,
    )
    print(f"[log] Writing training log to {log_path}")
    print("[log] Command:", " ".join(sys.argv))
    return f


class _JSONLMetricsCallback(TrainerCallback):
    def __init__(self, fh):
        self._fh = fh

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self._fh is None:
            return control
        rec = {"step": int(getattr(state, "global_step", 0)), "time": time.time()}
        for k, v in dict(logs).items():
            # Make values JSON-serializable.
            if isinstance(v, (int, float, str, bool)) or v is None:
                rec[k] = v
            else:
                try:
                    rec[k] = float(v)
                except Exception:
                    rec[k] = str(v)
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return control


def _expand_jsonl_inputs(path_or_glob: Optional[str]) -> List[Path]:
    spec = (path_or_glob or "").strip()
    if not spec:
        auto = sorted(Path(".").glob("itssm_lcb_v*.jsonl"))
        if not auto:
            raise FileNotFoundError("No itssm_lcb_v*.jsonl found in CWD; pass --tasks-jsonl explicitly.")
        return [p.resolve() for p in auto]

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: List[Path] = []
    for part in parts:
        p = Path(part)
        if p.exists() and p.is_dir():
            out.extend([c.resolve() for c in sorted(p.glob("itssm_lcb_v*.jsonl"))])
            continue
        if any(ch in part for ch in ["*", "?", "["]):
            out.extend([Path(m).resolve() for m in sorted(glob.glob(part)) if Path(m).is_file()])
            continue
        if p.exists() and p.is_file():
            out.append(p.resolve())
            continue
        raise FileNotFoundError(f"JSONL input not found: {part}")

    if not out:
        raise FileNotFoundError("No JSONL inputs remain.")

    seen: set[str] = set()
    uniq: List[Path] = []
    for p in out:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _load_jsonl(paths: Sequence[Path], limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def _extract_prompt_fields(example: Dict[str, Any]) -> Tuple[str, str, str]:
    problem = (example.get("question_content") or example.get("problem") or "").strip()
    starter = (example.get("starter_code") or "").strip()
    draft = (example.get("draft_code") or example.get("initial_proof_term") or "").strip()
    return problem, starter, draft


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT warm-start for DeepSeek Prover as ITSSM critic (combined tags).")
    ap.add_argument("--model-name-or-path", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    ap.add_argument("--output-dir", type=str, default="sft_deepseek_prover_out_critic")

    ap.add_argument("--tasks-jsonl", type=str, default=None, help="Path/glob/dir/comma-list (defaults to itssm_lcb_v*.jsonl in CWD)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    # Tokenization lengths
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--max-completion-tokens", type=int, default=512)
    ap.add_argument(
        "--min-present-target-tags",
        type=int,
        default=3,
        help="Minimum number of non-empty targets among {subgoals,gap_analysis,checklist} required to keep a row.",
    )

    # Optim / schedule
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Default reduced to avoid overfitting on small ITSSM JSONLs; increase if you have more data.",
    )
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-ratio", type=float, default=0.05, help="Holdout ratio for eval_loss logging (0 disables).")
    ap.add_argument("--eval-steps", type=int, default=50)

    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")

    # PEFT / quant
    ap.add_argument("--peft", choices=["lora", "qlora"], default="qlora")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"])
    ap.add_argument("--lora-target-modules", type=str, default="all-linear")
    ap.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true")

    # Saving
    ap.add_argument("--save-strategy", choices=["no", "steps"], default="no")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--no-save-final", action="store_true", help="Do not save final model/tokenizer to --output-dir")

    ap.add_argument("--log-dir", type=str, default="logs")
    ap.add_argument("--log-file", type=str, default="", help="Optional explicit log file path (default: auto under --log-dir)")
    ap.add_argument("--no-log-file", action="store_true", help="Disable saving logs to a file")
    ap.add_argument("--metrics-jsonl", type=str, default="", help="Optional JSONL metrics file path (default: auto under --log-dir)")
    ap.add_argument("--no-metrics-jsonl", action="store_true", help="Disable saving structured metrics JSONL")

    args = ap.parse_args()

    log_fh = None
    if not args.no_log_file:
        if args.log_file.strip():
            log_path = Path(args.log_file).expanduser().resolve()
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = Path(args.log_dir) / f"sft_prover_{ts}.log"
        log_fh = _setup_logging(log_path)
    else:
        _setup_logging(None)

    metrics_fh = None
    metrics_cb = None
    if not args.no_metrics_jsonl:
        try:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        except Exception:
            rank = 0
        if rank == 0:
            if args.metrics_jsonl.strip():
                mp = Path(args.metrics_jsonl).expanduser().resolve()
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                mp = Path(args.log_dir) / f"sft_metrics_{ts}.jsonl"
            mp.parent.mkdir(parents=True, exist_ok=True)
            metrics_fh = mp.open("w", encoding="utf-8", buffering=1)
            print(f"[metrics] Writing JSONL metrics to {mp}")
            metrics_cb = _JSONLMetricsCallback(metrics_fh)

    # bitsandbytes 4-bit GEMMs can fail with BF16 activations on some CUDA/cuBLAS stacks.
    # For QLoRA, force FP16 to avoid `CUBLAS_STATUS_NOT_SUPPORTED` failures.
    if args.peft == "qlora" and args.bf16:
        print("[warn] --bf16 + --peft qlora is not supported on this setup; forcing fp16 for QLoRA.")
        args.bf16 = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    paths = _expand_jsonl_inputs(args.tasks_jsonl)
    print(f"[data] Loading {len(paths)} JSONL(s): " + ", ".join([p.name for p in paths]))
    raw = _load_jsonl(paths, limit=args.limit)
    if not raw:
        raise SystemExit("No training rows loaded.")
    print(f"[data] Loaded {len(raw)} rows")

    # Build text pairs.
    examples: List[Dict[str, Any]] = []
    skipped_empty = 0
    skipped_low_quality = 0
    for ex in raw:
        problem, starter, draft = _extract_prompt_fields(ex)
        prompt = build_prover_stage2_prompt(problem, draft)
        subgoals_list, gap, checklist = extract_itssm_targets(ex)
        present_tags = int(bool(subgoals_list)) + int(bool(str(gap or "").strip())) + int(bool(str(checklist or "").strip()))
        if present_tags <= 0:
            skipped_empty += 1
            continue
        if present_tags < int(args.min_present_target_tags):
            skipped_low_quality += 1
            continue
        completion = (
            "<subgoal>\n"
            + "\n".join([s for s in subgoals_list if s.strip()]).strip()
            + "\n</subgoal>\n"
            + "<gap_analysis>\n"
            + _as_bullets(gap).strip()
            + "\n</gap_analysis>\n"
            + "<checklist>\n"
            + _as_bullets(checklist).strip()
            + "\n</checklist>\n"
        )
        examples.append({"prompt": prompt, "completion": completion, "starter_code": starter})
    if skipped_empty:
        print(f"[data] Skipped {skipped_empty} rows with empty targets after fallback parsing")
    if skipped_low_quality:
        print(
            f"[data] Skipped {skipped_low_quality} rows with < {int(args.min_present_target_tags)} "
            "non-empty target tags (quality filter)"
        )
    if not examples:
        raise SystemExit(
            "No SFT rows left after target fallback + quality filtering. "
            "Lower --min-present-target-tags or regenerate dataset targets."
        )

    from datasets import Dataset  # type: ignore

    ds = Dataset.from_list(examples)
    if len(ds) < 50:
        print(f"[warn] Very small SFT dataset (n={len(ds)}). Consider regenerating ITSSM targets or lowering --max-steps.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_prompt = int(args.max_prompt_tokens)
    max_comp = int(args.max_completion_tokens)
    max_len = max_prompt + max_comp + 2
    trunc_prompt = 0
    trunc_comp = 0

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        nonlocal trunc_prompt, trunc_comp
        input_ids_list: List[List[int]] = []
        attn_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            p0 = tokenizer(prompt, add_special_tokens=False, truncation=False)
            c0 = tokenizer(completion, add_special_tokens=False, truncation=False)
            if len(p0["input_ids"]) > max_prompt:
                trunc_prompt += 1
            if len(c0["input_ids"]) > max_comp:
                trunc_comp += 1
            p = {"input_ids": p0["input_ids"][:max_prompt]}
            c = {"input_ids": c0["input_ids"][:max_comp]}
            input_ids = p["input_ids"] + c["input_ids"] + [tokenizer.eos_token_id]
            labels = [-100] * len(p["input_ids"]) + c["input_ids"] + [tokenizer.eos_token_id]
            attn = [1] * len(input_ids)
            # Right pad
            pad_len = max(0, max_len - len(input_ids))
            if pad_len:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
                attn = attn + [0] * pad_len
            else:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                attn = attn[:max_len]
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attn_list.append(attn)
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attn_list}

    ds_tok = ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
    if len(ds_tok) > 0:
        print(
            f"[data] Token truncation stats: prompts={trunc_prompt}/{len(ds_tok)} "
            f"completions={trunc_comp}/{len(ds_tok)}"
        )
    eval_ratio = float(args.eval_ratio)
    if eval_ratio > 0.0 and len(ds_tok) >= 50:
        split = ds_tok.train_test_split(test_size=min(0.5, max(0.01, eval_ratio)), seed=int(args.seed))
        train_ds = split["train"]
        eval_ds = split["test"]
        print(f"[data] SFT split: train={len(train_ds)} eval={len(eval_ds)} (eval_ratio={eval_ratio})")
    else:
        train_ds = ds_tok
        eval_ds = None

    # Policy model + PEFT
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    peft_config = None
    from peft import LoraConfig, TaskType  # type: ignore

    if args.lora_target_modules.strip().lower() == "all-linear":
        target_modules: Any = "all-linear"
    else:
        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    device_map = None
    quantization_config = None
    if args.peft == "qlora":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device_map = {"": local_rank}
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    if args.peft == "qlora":
        try:
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except Exception:
            pass

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # Wrap with PEFT (Trainer will handle saving adapter weights).
    from peft import get_peft_model  # type: ignore

    model = get_peft_model(model, peft_config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targs_kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.grad_accum),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "max_steps": int(args.max_steps),
        "logging_steps": int(args.logging_steps),
        "eval_steps": int(args.eval_steps),
        "save_strategy": str(args.save_strategy),
        "save_steps": int(args.save_steps),
        "save_total_limit": 1,
        "bf16": bool(args.bf16),
        "fp16": not bool(args.bf16),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_pin_memory": True,
    }
    eval_mode = "steps" if eval_ds is not None else "no"
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = ta_sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ta_params.values())
    if not accepts_var_kwargs:
        targs_kwargs = {k: v for k, v in targs_kwargs.items() if k in ta_params}

    eval_keys: List[str] = []
    if "eval_strategy" in ta_params:
        eval_keys.append("eval_strategy")
    if "evaluation_strategy" in ta_params:
        eval_keys.append("evaluation_strategy")
    if not eval_keys:
        eval_keys = ["eval_strategy", "evaluation_strategy"]

    targs = None
    last_eval_exc: Optional[TypeError] = None
    for eval_key in eval_keys:
        cand = dict(targs_kwargs)
        cand[eval_key] = eval_mode
        try:
            targs = TrainingArguments(**cand)
            break
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg and ("evaluation_strategy" in msg or "eval_strategy" in msg):
                last_eval_exc = exc
                continue
            raise
    if targs is None:
        if last_eval_exc is not None:
            raise last_eval_exc
        raise RuntimeError("Failed to construct TrainingArguments with eval strategy compatibility fallback.")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[metrics_cb] if metrics_cb is not None else None,
    )
    trainer.train()

    if not args.no_save_final:
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        (out_dir / "base_model.json").write_text(
            json.dumps({"base_model": args.model_name_or_path, "peft": args.peft, "script": Path(__file__).name}, indent=2),
            encoding="utf-8",
        )

    if log_fh is not None:
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass
    if metrics_fh is not None:
        try:
            metrics_fh.flush()
            metrics_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
