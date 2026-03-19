#!/usr/bin/env python3
"""
RL fine-tuning for DeepSeek Prover (PPO or GRPO) on code-generation tasks with:
  - Dense step reward: subgoal alignment
  - Sparse terminal reward: unit test pass/fail

This script is designed to plug into the existing prompt/task plumbing in this repo:
  - Imports BigCodeBench task loading from:
      `fdg_approach_qwen3_inter_refined_reflex_robust_bigcodebench.py`
  - Reuses BigCodeBench's own sandboxed executor when available:
      `bigcodebench.eval.untrusted_check`

Notes
-----
- Default algorithm is PPO (via `trl.PPOTrainer`). A GRPO path is provided if
  `trl.GRPOTrainer` exists in your installed TRL version.
- For subgoal alignment, you can choose:
    (a) embedding cosine similarity (default; uses a frozen encoder from `transformers`)
    (b) a frozen reward model (sequence classifier) that scores (problem, step)
- Running arbitrary generated code is dangerous. For BigCodeBench tasks we rely on
  BigCodeBench's `untrusted_check` (multiprocessing + `reliability_guard`) to reduce risk.
  For non-BigCodeBench tasks, a minimal subprocess-based executor is provided as a placeholder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Disable TensorFlow/Flax in Transformers (avoids common TF/protobuf conflicts).
# Transformers uses USE_TF/USE_FLAX/USE_TORCH for backend selection.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
# PEFT import can fail in some environments due to TF/protobuf issues; pure-python protobuf is slower but robust.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:  # Optional: only for a formal Gym API.
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_callback import TrainerCallback
from transformers import BitsAndBytesConfig

# -----------------------------
# Repo integration (task loader)
# -----------------------------

def _load_tasks_from_repo_bigcodebench(
    split: str,
    subset: str,
    limit: Optional[int],
    start_index: int,
    end_index: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Import the existing BigCodeBench loader to keep task formatting consistent with the repo.
    """

    from fdg_approach_qwen3_inter_refined_reflex_robust_bigcodebench import load_bigcodebench_tasks

    tasks = load_bigcodebench_tasks(
        split=split,
        subset=subset,
        include_tests=True,
        include_canonical_solution=False,
    )
    if end_index is None:
        end_index = len(tasks)
    tasks = tasks[start_index:end_index]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def _load_tasks_from_livecodebench(
    versions: Sequence[str],
    limit: Optional[int],
    start_index: int,
    end_index: Optional[int],
    hf_cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    include_private_tests: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load LiveCodeBench code_generation_lite tasks for multiple `version_tag`s (e.g. v1..v5).

    Returns dicts compatible with this RL script:
      - question_id, question_content, starter_code
      - input_output (JSON string with inputs/outputs/fn_name), used by LiveCodeBench executor
      - metadata (parsed)
      - lcb_version (string)
    """

    lcb_root = Path(__file__).parent / "LiveCodeBench"
    if lcb_root.exists():
        sys.path.insert(0, str(lcb_root))

    try:
        from datasets import Dataset, concatenate_datasets  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("`datasets` is required to load LiveCodeBench.") from exc

    versions_norm = [v.strip() for v in versions if v and v.strip()]

    # Common request: use v1-v5 problems as the RL dataset.
    # LiveCodeBench dataset supports configs like `release_v5` (= union of v1..v5).
    if set(versions_norm) == {"v1", "v2", "v3", "v4", "v5"}:
        versions_norm = ["release_v5"]
    elif set(versions_norm) == {"v1", "v2", "v3", "v4", "v5", "v6"}:
        versions_norm = ["release_latest"]

    def _detect_global_datasets_cache_dir() -> Path:
        # In this Codex sandbox, we can typically read from ~/.cache but cannot write locks there.
        # We'll read cached arrow files directly (no locking) to stay compatible with workspace-only writes.
        env_cache = os.environ.get("HF_DATASETS_CACHE")
        if env_cache:
            return Path(env_cache).expanduser()
        return (Path.home() / ".cache" / "huggingface" / "datasets").expanduser()

    def _load_from_arrow_config(cache_root: Path, config_name: str) -> "Dataset":
        base = cache_root / "livecodebench___code_generation_lite" / config_name
        if not base.exists():
            raise FileNotFoundError(str(base))
        # Expect `<config>/0.0.0/<hash>/code_generation_lite-test*.arrow`, but ignore lock files.
        arrow_files_all = sorted(base.rglob("code_generation_lite-test*.arrow"))
        if not arrow_files_all:
            raise FileNotFoundError(f"No arrow shards found under {base}")

        by_parent: Dict[Path, List[Path]] = {}
        for p in arrow_files_all:
            by_parent.setdefault(p.parent, []).append(p)
        # Prefer the parent dir with most shards (release_latest is split), break ties by mtime.
        parent, arrow_files = max(
            by_parent.items(),
            key=lambda kv: (len(kv[1]), kv[0].stat().st_mtime),
        )
        arrow_files = sorted(arrow_files)
        parts = [Dataset.from_file(str(p)) for p in arrow_files]
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    cache_root = _detect_global_datasets_cache_dir()

    # We only rely on configs already present in the cache.
    # - `release_latest` contains all tasks (v1..v6).
    # - `release_latest-version_tag=v6` (created by some scripts in this repo) contains the v6-only slice.
    #   We can derive v1..v5 as (release_latest \ v6).
    def _load_release_latest() -> "Dataset":
        return _load_from_arrow_config(cache_root, "release_latest")

    def _load_v6_only() -> "Dataset":
        return _load_from_arrow_config(cache_root, "release_latest-version_tag=v6")

    datasets_by_version: Dict[str, "Dataset"] = {}
    need_release_latest = True  # offline loader always uses release_latest as the source of truth
    if need_release_latest:
        datasets_by_version["release_latest"] = _load_release_latest()

    # v6-only cached slice is used to determine the boundary between v1..v5 and v6.
    # If missing, we fall back to `v6_len = 175` as a best-effort default.
    try:
        datasets_by_version["v6_only"] = _load_v6_only()
    except Exception:
        datasets_by_version["v6_only"] = None  # type: ignore[assignment]

    def _derive_partitions(ds_all: "Dataset", ds_v6: Optional["Dataset"]) -> Dict[str, slice]:
        """
        Derive per-version contiguous slices in the *release_latest order*.

        In the cached `release_latest` used in this repo, tasks are stored as concatenation of (v1..v6) files, so
        each version corresponds to a contiguous segment. We use the cached v6-only length to find the boundary.
        """

        total = len(ds_all)
        v6_len = len(ds_v6) if ds_v6 is not None else 175
        pre_v6 = total - v6_len
        if pre_v6 <= 0:
            raise RuntimeError(f"Invalid cache: total={total}, v6_len={v6_len}")

        # Prefer equal-sized chunks for v1..v5. If not divisible, distribute the remainder.
        base = pre_v6 // 5
        rem = pre_v6 % 5
        sizes = [base + (1 if i < rem else 0) for i in range(5)]
        starts = [0]
        for s in sizes[:-1]:
            starts.append(starts[-1] + s)

        slices: Dict[str, slice] = {}
        for i, (st, sz) in enumerate(zip(starts, sizes), start=1):
            slices[f"v{i}"] = slice(st, st + sz)
            slices[f"release_v{i}"] = slice(0, st + sz)
        slices["release_v5"] = slice(0, pre_v6)
        slices["v6"] = slice(pre_v6, total)
        slices["release_v6"] = slice(0, total)
        slices["release_latest"] = slice(0, total)
        return slices

    all_tasks: List[Dict[str, Any]] = []
    global_start = max(0, start_index)
    global_end = (10**18) if end_index is None else max(global_start, end_index)
    remaining = None if limit is None else max(0, limit)
    global_cursor = 0

    ds_all = datasets_by_version.get("release_latest")
    if ds_all is None:
        raise RuntimeError("Missing cached LiveCodeBench `release_latest` dataset.")
    ds_v6 = datasets_by_version.get("v6_only")
    if ds_v6 is not None and ds_v6 is not None and not isinstance(ds_v6, Dataset):
        ds_v6 = None
    partitions = _derive_partitions(ds_all, ds_v6)  # maps version tag -> slice in ds_all

    for version in versions_norm:
        if version not in partitions:
            raise RuntimeError(
                f"Unknown LiveCodeBench version tag: {version}. "
                "Supported offline tags: v1..v6, release_v1..release_v6, release_latest."
            )
        ds = ds_all.select(range(partitions[version].start or 0, partitions[version].stop or len(ds_all)))
        tag = version

        total = len(ds)
        if global_cursor >= global_end:
            break

        # Choose which rows from this `ds` fall into the global [start, end) window.
        local_indices: List[int] = []
        for i in range(total):
            g = global_cursor + i
            if g < global_start:
                continue
            if g >= global_end:
                break
            if remaining is not None and remaining <= 0:
                break
            local_indices.append(i)
            if remaining is not None:
                remaining -= 1

        global_cursor += total
        if not local_indices:
            continue

        ds = ds.select(local_indices)

        def _decode_private_tests(raw: str) -> list[dict]:
            import base64
            import pickle
            import zlib

            try:
                return json.loads(raw)
            except Exception:
                return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))

        for row in ds:
            metadata = json.loads(row.get("metadata") or "{}")
            fn_name = metadata.get("func_name", None)

            public_tests = json.loads(row.get("public_test_cases") or "[]")
            tests = list(public_tests)
            if include_private_tests:
                priv_raw = row.get("private_test_cases") or "[]"
                tests.extend(_decode_private_tests(priv_raw))

            inputs = [t.get("input", "") for t in tests]
            outputs = [t.get("output", "") for t in tests]
            input_output = json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": fn_name})

            all_tasks.append(
                {
                    "question_id": f"{tag}/{row.get('question_id')}",
                    "question_title": row.get("question_title", ""),
                    "question_content": row.get("question_content", ""),
                    "starter_code": row.get("starter_code", ""),
                    "input_output": input_output,
                    "metadata": metadata,
                    "lcb_version": tag,
                }
            )
    return all_tasks


def _expand_jsonl_inputs(path_or_glob: str) -> List[Path]:
    """
    Expand a JSONL input specification into concrete file paths.

    Supports:
      - a single file path
      - a directory (loads `itssm_lcb_v*.jsonl` if present, else `*.jsonl`)
      - a glob pattern (e.g. `itssm_lcb_v*.jsonl`)
      - a comma-separated list of any of the above
    """

    spec = (path_or_glob or "").strip()
    if not spec:
        return []

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    paths: List[Path] = []
    for part in parts:
        p = Path(part)
        if p.exists() and p.is_dir():
            itssm = sorted(p.glob("itssm_lcb_v*.jsonl"))
            candidates = itssm if itssm else sorted(p.glob("*.jsonl"))
            paths.extend([c.resolve() for c in candidates])
            continue

        # If it looks like a glob, expand it (even if the shell didn't).
        if any(ch in part for ch in ["*", "?", "["]):
            for m in sorted(glob.glob(part)):
                mp = Path(m)
                if mp.exists() and mp.is_file():
                    paths.append(mp.resolve())
            continue

        if p.exists() and p.is_file():
            paths.append(p.resolve())
            continue

        raise FileNotFoundError(f"JSONL input not found: {part}")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _load_tasks_from_jsonl(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    """
    JSONL schema (minimal):
      {"task_id": "...", "problem": "...", "starter_code": "...", "test": "...", "entry_point": "..."}

    Optional:
      - "subgoals": ["...", "..."]  (preferred for dense reward)
      - "complete_prompt": "..."    (used to execute tests like BigCodeBench does)
    """

    tasks: List[Dict[str, Any]] = []
    paths = _expand_jsonl_inputs(path)
    if not paths:
        raise FileNotFoundError("Empty JSONL path specification.")
    print(f"[jsonl] Loading {len(paths)} file(s)")
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tasks.append(json.loads(line))
                if limit is not None and len(tasks) >= limit:
                    return tasks
    return tasks


# -----------------------------
# Text parsing and prompt format
# -----------------------------

_TAG_SUBGOAL = "subgoal"
_TAG_CODE = "code"
_TAG_FINAL = "final_code"
_TAG_GAP = "gap_analysis"
_TAG_TYPE = "type_analysis"


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "").strip()


def _extract_code_fence(text: str) -> str:
    # Best-effort extraction of the first fenced code block.
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def build_step_prompt(
    problem_text: str,
    starter_code: str,
    code_so_far: str,
    generated_subgoals: Sequence[str],
    step_idx: int,
    n_subgoal_steps: int,
) -> str:
    """
    State = (problem, code_so_far, generated_subgoals, step_idx).
    Action = generate either the next subgoal or the final code.
    """

    header = (
        "You are DeepSeek Prover acting as a code-generation policy.\n"
        "You solve the problem by writing a short list of natural-language subgoals, then producing code.\n"
        "Follow the required output format exactly.\n"
    )

    prompt = header
    prompt += "\n### Problem\n" + (problem_text or "").strip() + "\n"

    if starter_code:
        prompt += "\n### Starter Code\n```python\n" + starter_code.strip() + "\n```\n"

    if code_so_far:
        prompt += "\n### Code So Far\n```python\n" + code_so_far.strip() + "\n```\n"

    if generated_subgoals:
        prompt += "\n### Subgoals So Far\n"
        for i, sg in enumerate(generated_subgoals, start=1):
            prompt += f"{i}. {sg.strip()}\n"

    if step_idx < n_subgoal_steps:
        prompt += (
            f"\n### Task\nWrite subgoal #{step_idx + 1} as one sentence.\n"
            "Return ONLY:\n"
            f"<{_TAG_SUBGOAL}>...one sentence...</{_TAG_SUBGOAL}>\n"
        )
    else:
        prompt += (
            "\n### Task\nWrite the final Python solution implementing the subgoals.\n"
            "Return ONLY:\n"
            f"<{_TAG_FINAL}>```python\\n...code...\\n```</{_TAG_FINAL}>\n"
        )
    return prompt


def default_intended_subgoals(problem_text: str, max_n: int) -> List[str]:
    """
    Fallback "intended subgoals" extractor when the dataset doesn't provide subgoals.
    This is intentionally simple; for serious training you should provide labeled subgoals
    or precompute them with a frozen teacher model and store them in your JSONL.
    """

    text = (problem_text or "").strip()
    if not text:
        return []

    # Prefer explicit bullets/numbering if present.
    bullets: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(\d+\.|- |\* )", line):
            s = re.sub(r"^(\d+\.|- |\* )\s*", "", line).strip()
            if s:
                bullets.append(s)
    bullets = [b for b in bullets if len(b) >= 12]
    if bullets:
        return bullets[:max_n]

    # Otherwise: take the first few sentences as pseudo-subgoals.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 12]
    return sentences[:max_n]


# -----------------------------
# Dense reward: subgoal alignment
# -----------------------------

class SubgoalAlignmentScorer:
    def score(self, *, problem: str, generated_step: str, intended_subgoal: str) -> float:
        raise NotImplementedError


class EmbeddingCosineScorer(SubgoalAlignmentScorer):
    """
    Frozen encoder model scorer using cosine similarity.

    Recommended checkpoints (examples):
      - intfloat/e5-small-v2
      - sentence-transformers/all-MiniLM-L6-v2  (works via transformers too)
    """

    def __init__(self, model_name_or_path: str, device: torch.device, max_length: int = 256):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _embed(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs)
        last_hidden = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, p=2, dim=1)

    @torch.no_grad()
    def score(self, *, problem: str, generated_step: str, intended_subgoal: str) -> float:
        a = (generated_step or "").strip()
        b = (intended_subgoal or "").strip()
        if not a or not b:
            return 0.0
        emb = self._embed([a, b])
        sim = float((emb[0] * emb[1]).sum().clamp(-1, 1).item())
        # map cosine [-1, 1] -> [0, 1]
        return 0.5 * (sim + 1.0)


class FrozenRewardModelScorer(SubgoalAlignmentScorer):
    """
    Frozen reward model for (problem, step) -> score.

    Expected: a `AutoModelForSequenceClassification` that produces a single scalar logit.
    Reward is `sigmoid(logit)` in [0, 1].
    """

    def __init__(self, model_name_or_path: str, device: torch.device, max_length: int = 512):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score(self, *, problem: str, generated_step: str, intended_subgoal: str) -> float:
        step = (generated_step or "").strip()
        target = (intended_subgoal or "").strip()
        if not step:
            return 0.0
        # Include the "intended_subgoal" as a target description (works for pairwise RMs too).
        prompt = f"Problem:\n{problem.strip()}\n\nIntended subgoal:\n{target}\n\nGenerated step:\n{step}\n"
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs)
        logit = out.logits.squeeze()
        return float(torch.sigmoid(logit).clamp(0, 1).item())


# -----------------------------
# Sparse reward: unit test pass/fail
# -----------------------------

@dataclass
class TestResult:
    passed: bool
    status: str
    details: Dict[str, Any]


class CodeExecutor:
    def run_tests(self, *, task: Dict[str, Any], solution_code: str) -> TestResult:
        raise NotImplementedError


class BigCodeBenchExecutor(CodeExecutor):
    def __init__(
        self,
        *,
        max_as_limit: int = 30 * 1024,
        max_data_limit: int = 30 * 1024,
        max_stack_limit: int = 10,
        min_time_limit: float = 1.0,
        gt_time_limit: float = 2.0,
    ):
        self.max_as_limit = max_as_limit
        self.max_data_limit = max_data_limit
        self.max_stack_limit = max_stack_limit
        self.min_time_limit = min_time_limit
        self.gt_time_limit = gt_time_limit

        try:
            from bigcodebench.eval import untrusted_check  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "BigCodeBench executor requested but bigcodebench is not importable.\n"
                "Install it with:\n"
                "  cd lean_gen/bigcodebench && pip install -e .\n"
            ) from exc

    def run_tests(self, *, task: Dict[str, Any], solution_code: str) -> TestResult:
        from bigcodebench.eval import untrusted_check

        test_code = (task.get("test") or "").strip()
        entry_point = (task.get("entry_point") or "").strip()

        # BigCodeBench evaluates `complete_prompt + "\n" + solution`.
        complete_prompt = (task.get("complete_prompt") or task.get("question_content") or "").strip()
        full_code = (complete_prompt + "\n" + (solution_code or "")).strip() + "\n"

        status, details = untrusted_check(
            full_code,
            test_code,
            entry_point,
            self.max_as_limit,
            self.max_data_limit,
            self.max_stack_limit,
            self.min_time_limit,
            self.gt_time_limit,
        )
        passed = status == "pass"
        return TestResult(passed=passed, status=status, details=details)


class LiveCodeBenchExecutor(CodeExecutor):
    """
    Uses LiveCodeBench's own test runner (`lcb_runner.evaluation.testing_util.run_test`)
    wrapped in a subprocess for better isolation.
    """

    def __init__(self, timeout_s: int = 6):
        self.timeout_s = timeout_s
        lcb_root = Path(__file__).parent / "LiveCodeBench"
        if lcb_root.exists():
            sys.path.insert(0, str(lcb_root))
        try:
            from lcb_runner.evaluation.testing_util import run_test  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LiveCodeBench runner not importable. Ensure `LiveCodeBench/` exists in this repo."
            ) from exc

    def run_tests(self, *, task: Dict[str, Any], solution_code: str) -> TestResult:
        import multiprocessing

        from lcb_runner.evaluation.testing_util import run_test  # type: ignore

        raw_input_output = task.get("input_output", "")
        if isinstance(raw_input_output, dict):
            try:
                raw_input_output = json.dumps(raw_input_output, ensure_ascii=False)
            except Exception:
                raw_input_output = ""
        elif raw_input_output is None:
            raw_input_output = ""
        elif not isinstance(raw_input_output, str):
            raw_input_output = str(raw_input_output)

        if not raw_input_output:
            return TestResult(passed=False, status="no_tests", details={"ALL": "Missing input_output"})
        try:
            json.loads(raw_input_output)
        except Exception as exc:
            return TestResult(
                passed=False,
                status="invalid_tests",
                details={"meta": {"error_code": -4, "error_message": f"Invalid input_output JSON: {exc}"}},
            )

        sample = {"input_output": raw_input_output}

        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)

        def _worker(conn) -> None:
            try:
                res, meta = run_test(sample, test=solution_code, debug=False, timeout=self.timeout_s)
                conn.send({"res": res, "meta": meta})
            except BaseException as e:
                conn.send({"res": [-4], "meta": {"error": repr(e), "error_code": -4}})
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        p = multiprocessing.Process(target=_worker, args=(child_conn,))
        p.start()
        try:
            child_conn.close()
        except Exception:
            pass
        p.join(timeout=(self.timeout_s + 1) * 3 + 5)
        if p.is_alive():
            p.kill()
            p.join(timeout=1)

        if not parent_conn.poll(timeout=0.2):
            return TestResult(passed=False, status="timeout", details={"ALL": "global timeout"})

        try:
            pkt = parent_conn.recv()
        except Exception:
            return TestResult(passed=False, status="timeout", details={"ALL": "missing worker result"})
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass
        res = list(pkt.get("res") or [])
        raw_meta = pkt.get("meta") if isinstance(pkt, dict) else {}
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        passed = bool(res) and all(x is True for x in res)
        return TestResult(passed=passed, status="pass" if passed else "fail", details={"results": res, "meta": meta})


class SubprocessPythonExecutor(CodeExecutor):
    """
    Placeholder executor for non-BigCodeBench tasks.
    This is NOT a hardened sandbox; use Docker/Firejail/nsjail in production.
    """

    def __init__(self, timeout_s: int = 10):
        self.timeout_s = timeout_s

    def run_tests(self, *, task: Dict[str, Any], solution_code: str) -> TestResult:
        test_code = (task.get("test") or "").strip()
        if not test_code:
            return TestResult(passed=False, status="no_tests", details={"ALL": "No tests provided"})

        with tempfile.TemporaryDirectory(prefix="rl_exec_") as tmp:
            tmp_path = Path(tmp)
            sol_path = tmp_path / "solution.py"
            test_path = tmp_path / "test_solution.py"
            sol_path.write_text(solution_code or "", encoding="utf-8")
            test_path.write_text(test_code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", str(test_path)],
                    cwd=str(tmp_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                return TestResult(passed=False, status="timeout", details={"ALL": "Timeout"})
            return TestResult(
                passed=proc.returncode == 0,
                status="pass" if proc.returncode == 0 else "fail",
                details={"stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]},
            )


# -----------------------------
# Gym-like environment (text)
# -----------------------------

_GymBase = gym.Env if gym is not None else object  # type: ignore[attr-defined]


class CodeGenEnv(_GymBase):
    """
    Gym-like environment:
      state: (problem, code_so_far, subgoals_so_far, step_idx)
      action: LLM-generated text (tagged)
      reward:
        - dense step reward: subgoal alignment for subgoal steps
        - sparse terminal: unit tests on final code
    """

    def __init__(
        self,
        *,
        task: Dict[str, Any],
        scorer: SubgoalAlignmentScorer,
        executor: CodeExecutor,
        n_subgoal_steps: int,
        dense_weight: float,
        terminal_weight: float,
        intended_subgoals: Optional[Sequence[str]] = None,
    ):
        self.task = task
        self.problem = (task.get("question_content") or task.get("problem") or "").strip()
        self.starter_code = (task.get("starter_code") or "").strip()
        self.scorer = scorer
        self.executor = executor
        self.n_subgoal_steps = n_subgoal_steps
        self.dense_weight = dense_weight
        self.terminal_weight = terminal_weight
        self._intended_subgoals = list(intended_subgoals or [])

        if gym is not None:  # type: ignore[truthy-bool]
            # Minimal spaces for compatibility; the policy itself is handled by the LLM/TRL.
            # `gymnasium` exposes `spaces.Text`; classic `gym` may not.
            if hasattr(gym.spaces, "Text"):  # type: ignore[attr-defined]
                self.action_space = gym.spaces.Text(max_length=8192)  # type: ignore[attr-defined]
                self.observation_space = gym.spaces.Dict(  # type: ignore[attr-defined]
                    {"prompt": gym.spaces.Text(max_length=8192)}  # type: ignore[attr-defined]
                )
            else:
                # Fallback: placeholder spaces.
                self.action_space = gym.spaces.Discrete(1)  # type: ignore[attr-defined]
                self.observation_space = gym.spaces.Dict(  # type: ignore[attr-defined]
                    {"prompt": gym.spaces.Discrete(1)}  # type: ignore[attr-defined]
                )

        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.step_idx = 0
        self.subgoals: List[str] = []
        self.code_so_far = self.starter_code
        return self._obs()

    def _obs(self) -> Dict[str, Any]:
        return {
            "prompt": build_step_prompt(
                problem_text=self.problem,
                starter_code=self.starter_code,
                code_so_far=self.code_so_far,
                generated_subgoals=self.subgoals,
                step_idx=self.step_idx,
                n_subgoal_steps=self.n_subgoal_steps,
            )
        }

    def step(self, action_text: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}

        intended = self._get_intended_subgoal(self.step_idx)
        dense = 0.0
        terminal = 0.0
        done = False

        if self.step_idx < self.n_subgoal_steps:
            subgoal = _extract_tag(action_text, _TAG_SUBGOAL) or action_text.strip()
            subgoal = re.sub(r"\s+", " ", subgoal).strip()
            if subgoal:
                self.subgoals.append(subgoal)
            dense = self.scorer.score(problem=self.problem, generated_step=subgoal, intended_subgoal=intended)
            self.step_idx += 1
            done = False
            info.update({"dense": dense, "subgoal": subgoal, "intended_subgoal": intended})
        else:
            final = _extract_tag(action_text, _TAG_FINAL) or _extract_tag(action_text, _TAG_CODE) or action_text
            code = _extract_code_fence(final)
            self.code_so_far = code.strip()
            info["code_chars"] = len(self.code_so_far)
            tr = self.executor.run_tests(task=self.task, solution_code=self.code_so_far)
            terminal = 1.0 if tr.passed else 0.0
            info.update({"test_status": tr.status, "test_details": tr.details})
            done = True

        reward = self.dense_weight * dense + self.terminal_weight * terminal
        return self._obs(), reward, done, info

    def _get_intended_subgoal(self, step_idx: int) -> str:
        # Priority: explicit dataset subgoals -> fallback extracted subgoals -> problem text.
        dataset_subgoals = self.task.get("subgoals")
        if isinstance(dataset_subgoals, list) and dataset_subgoals:
            if step_idx < len(dataset_subgoals):
                return str(dataset_subgoals[step_idx])
            return str(dataset_subgoals[-1])

        if not self._intended_subgoals:
            self._intended_subgoals = default_intended_subgoals(self.problem, max_n=max(1, self.n_subgoal_steps))
        if self._intended_subgoals:
            if step_idx < len(self._intended_subgoals):
                return self._intended_subgoals[step_idx]
            return self._intended_subgoals[-1]

        return self.problem


# -----------------------------
# PPO rollout + update loop
# -----------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batched(iterable: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield list(iterable[i : i + batch_size])


def _decode_response(tokenizer: Any, response_tensors: torch.Tensor) -> str:
    # response_tensors is (seq,) or (1, seq)
    if response_tensors.ndim == 2:
        response_tensors = response_tensors[0]
    return tokenizer.decode(response_tensors, skip_special_tokens=True)


def train_ppo(args: argparse.Namespace) -> None:
    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import TRL PPO components (PPOTrainer/PPOConfig). "
            "Install compatible versions, e.g.\n"
            "  pip install 'trl>=0.9' 'transformers>=4.41' accelerate\n"
            "If you see NumPy typing errors, try upgrading to NumPy 2.x or downgrading Transformers."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    _seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )

    model.to(device)
    ref_model.to(device)

    # Subgoal alignment scorer (frozen).
    if args.subgoal_reward == "embedding":
        scorer: SubgoalAlignmentScorer = EmbeddingCosineScorer(
            args.subgoal_reward_model, device=device, max_length=args.subgoal_reward_max_length
        )
    elif args.subgoal_reward == "rm":
        scorer = FrozenRewardModelScorer(
            args.subgoal_reward_model, device=device, max_length=args.subgoal_reward_max_length
        )
    else:
        raise ValueError("--subgoal-reward must be 'embedding' or 'rm'")

    # Executor (terminal tests).
    if args.executor == "bigcodebench":
        executor: CodeExecutor = BigCodeBenchExecutor(
            max_as_limit=args.max_as_limit,
            max_data_limit=args.max_data_limit,
            max_stack_limit=args.max_stack_limit,
            min_time_limit=args.min_time_limit,
            gt_time_limit=args.gt_time_limit,
        )
    elif args.executor == "livecodebench":
        executor = LiveCodeBenchExecutor(timeout_s=int(args.min_time_limit))
    else:
        executor = SubprocessPythonExecutor(timeout_s=int(args.min_time_limit))

    # Tasks.
    if args.dataset == "jsonl":
        if not args.tasks_jsonl:
            # Convenience: auto-detect ITSSM JSONLs (common in this repo).
            auto = sorted(Path(".").glob("itssm_lcb_v*.jsonl"))
            if auto:
                args.tasks_jsonl = ",".join([str(p) for p in auto])
                print(f"[jsonl] Auto-detected ITSSM dataset files: {args.tasks_jsonl}")
            else:
                raise RuntimeError("--dataset jsonl requires --tasks-jsonl (or provide itssm_lcb_v*.jsonl in CWD)")
        tasks = _load_tasks_from_jsonl(args.tasks_jsonl, limit=args.limit)
    elif args.dataset == "bigcodebench":
        tasks = _load_tasks_from_repo_bigcodebench(
            split=args.bcb_split,
            subset=args.bcb_subset,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
        )
    elif args.dataset == "livecodebench":
        versions = [v.strip() for v in args.lcb_versions.split(",") if v.strip()]
        tasks = _load_tasks_from_livecodebench(
            versions=versions,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
        )
    else:
        raise ValueError("--dataset must be one of: livecodebench, bigcodebench, jsonl")
    if not tasks:
        raise RuntimeError("No tasks loaded. Provide --tasks-jsonl or ensure BigCodeBench is available.")

    # PPO config (KL control is the main anti-forgetting knob).
    # - `ref_model` is the non-updated baseline policy.
    # - `target_kl` + `adaptive_kl_ctrl` adjust KL coefficient online to keep updates conservative.
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.rollout_batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        ppo_epochs=args.ppo_epochs,
        init_kl_coef=args.kl_coef,
        target=args.target_kl,
        adap_kl_ctrl=not args.disable_adaptive_kl,
        cliprange=args.cliprange,
        cliprange_value=args.cliprange_value,
        vf_coef=args.vf_coef,
        seed=args.seed,
        log_with=args.log_with,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=None,
        data_collator=None,
    )

    gen_kwargs = {
        "do_sample": True,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main loop: repeatedly sample tasks -> run environment steps -> PPO update.
    global_step = 0
    for update_idx in range(args.updates):
        batch_tasks = random.sample(tasks, k=min(args.episodes_per_update, len(tasks)))

        query_tensors: List[torch.Tensor] = []
        response_tensors: List[torch.Tensor] = []
        rewards: List[float] = []

        logs: Dict[str, float] = {"dense": 0.0, "terminal": 0.0, "pass_rate": 0.0, "episodes": 0.0}
        passes = 0
        episodes = 0

        for task in batch_tasks:
            env = CodeGenEnv(
                task=task,
                scorer=scorer,
                executor=executor,
                n_subgoal_steps=args.n_subgoal_steps,
                dense_weight=args.dense_weight,
                terminal_weight=args.terminal_weight,
            )
            obs = env.reset()
            done = False
            while not done:
                prompt = obs["prompt"]
                q = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens).input_ids
                q = q.to(device)[0]
                # TRL PPOTrainer expects a list of query tensors.
                r = ppo_trainer.generate(q.unsqueeze(0), **gen_kwargs)[0]
                action_text = _decode_response(tokenizer, r)

                obs, reward, done, info = env.step(action_text)
                query_tensors.append(q)
                response_tensors.append(r)
                rewards.append(float(reward))
                global_step += 1

                if "dense" in info:
                    logs["dense"] += float(info["dense"])
                if "test_status" in info:
                    terminal = 1.0 if info["test_status"] == "pass" else 0.0
                    logs["terminal"] += terminal
                    if terminal == 1.0:
                        passes += 1
                    episodes += 1

                if len(query_tensors) >= args.rollout_batch_size:
                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                    query_tensors, response_tensors, rewards = [], [], []
                    # minimal console logging (avoid heavy deps)
                    approx_kl = float(stats.get("ppo/kl", 0.0))
                    mean_reward = float(stats.get("ppo/mean_scores", 0.0))
                    print(
                        f"[update {update_idx+1}/{args.updates}] step={global_step} "
                        f"reward={mean_reward:.3f} kl={approx_kl:.4f}"
                    )

        # Flush remaining rollouts.
        if query_tensors:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            approx_kl = float(stats.get("ppo/kl", 0.0))
            mean_reward = float(stats.get("ppo/mean_scores", 0.0))
            print(
                f"[update {update_idx+1}/{args.updates}] step={global_step} "
                f"reward={mean_reward:.3f} kl={approx_kl:.4f} (flush)"
            )

        if episodes > 0:
            logs["pass_rate"] = passes / episodes
            logs["episodes"] = float(episodes)
        if max(1, global_step) > 0:
            logs["dense"] /= max(1.0, logs["episodes"] * max(1.0, float(args.n_subgoal_steps)))
            logs["terminal"] /= max(1.0, logs["episodes"])
        print(
            f"[update {update_idx+1}/{args.updates}] pass_rate={logs['pass_rate']:.3f} "
            f"dense={logs['dense']:.3f} terminal={logs['terminal']:.3f}"
        )

        if args.no_save_checkpoints:
            continue
        if (update_idx + 1) % args.save_every == 0 or (update_idx + 1) == args.updates:
            save_path = out_dir / f"checkpoint-update-{update_idx+1}"
            save_path.mkdir(parents=True, exist_ok=True)
            ppo_trainer.model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            (save_path / "rl_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
            print(f"Saved checkpoint to {save_path}")


def train_grpo(args: argparse.Namespace) -> None:
    """
    GRPO training (recommended in this repo environment).

    Key memory knob for 2xA6000:
      - Use `--peft qlora` (default) so the base model stays 4-bit and only LoRA adapters are trained.
      - Keep `--per-device-train-batch-size 1` and enable `--gradient-checkpointing`.
      - Keep `--num-generations` modest (e.g. 4) since it multiplies rollout compute and transient memory.
    """

    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Your TRL installation does not expose GRPOTrainer.") from exc

    _seed_everything(args.seed)

    reward_device = torch.device(args.subgoal_reward_device)
    if args.subgoal_reward == "embedding":
        scorer: SubgoalAlignmentScorer = EmbeddingCosineScorer(
            args.subgoal_reward_model, device=reward_device, max_length=args.subgoal_reward_max_length
        )
    elif args.subgoal_reward == "rm":
        scorer = FrozenRewardModelScorer(
            args.subgoal_reward_model, device=reward_device, max_length=args.subgoal_reward_max_length
        )
    else:
        raise ValueError("--subgoal-reward must be 'embedding' or 'rm'")

    if args.executor == "livecodebench":
        executor: CodeExecutor = LiveCodeBenchExecutor(timeout_s=int(args.min_time_limit))
    elif args.executor == "bigcodebench":
        executor = BigCodeBenchExecutor(
            max_as_limit=args.max_as_limit,
            max_data_limit=args.max_data_limit,
            max_stack_limit=args.max_stack_limit,
            min_time_limit=args.min_time_limit,
            gt_time_limit=args.gt_time_limit,
        )
    else:
        executor = SubprocessPythonExecutor(timeout_s=int(args.min_time_limit))

    # Load tasks
    if args.dataset == "jsonl":
        if not args.tasks_jsonl:
            auto = sorted(Path(".").glob("itssm_lcb_v*.jsonl"))
            if auto:
                args.tasks_jsonl = ",".join([str(p) for p in auto])
                print(f"[jsonl] Auto-detected ITSSM dataset files: {args.tasks_jsonl}")
            else:
                raise RuntimeError("--dataset jsonl requires --tasks-jsonl (or provide itssm_lcb_v*.jsonl in CWD)")
        raw_tasks = _load_tasks_from_jsonl(args.tasks_jsonl, limit=args.limit)
    elif args.dataset == "bigcodebench":
        raw_tasks = _load_tasks_from_repo_bigcodebench(
            split=args.bcb_split,
            subset=args.bcb_subset,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
        )
    else:
        versions = [v.strip() for v in args.lcb_versions.split(",") if v.strip()]
        raw_tasks = _load_tasks_from_livecodebench(
            versions=versions,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
            hf_cache_dir=args.hf_cache_dir,
            include_private_tests=args.lcb_include_private_tests,
        )
    if not raw_tasks:
        raise RuntimeError("No tasks loaded.")

    num_generations = args.num_generations if args.num_generations is not None else args.grpo_group_size

    def make_prompt(task: Dict[str, Any]) -> str:
        """
        ITSSM-style prompt for training the *prover* to output:
          - subgoals (key invariants)
          - gap analysis
          - revised code

        If `draft_code` is present in the JSONL record, it is provided as the candidate implementation.
        """

        problem = (task.get("question_content") or task.get("problem") or "").strip()
        starter = (task.get("starter_code") or "").strip()
        draft = (task.get("draft_code") or "").strip()

        prompt = (
            "You are DeepSeek Prover. You analyze a candidate Python implementation, "
            "identify key invariants/subgoals and gaps, then produce a corrected solution.\n\n"
            "### Problem Description\n" + problem + "\n"
        )
        if starter:
            prompt += "\n### Starter Code\n```python\n" + starter + "\n```\n"
        if draft:
            prompt += "\n### Candidate Implementation (Python)\n```python\n" + draft + "\n```\n"

        prompt += (
            "\nReturn ONLY the following tags:\n"
            f"<{_TAG_SUBGOAL}>one subgoal per line</{_TAG_SUBGOAL}>\n"
            f"<{_TAG_GAP}>concise gap analysis bullets</{_TAG_GAP}>\n"
            f"<{_TAG_FINAL}>```python\\n...revised code...\\n```</{_TAG_FINAL}>\n"
        )
        return prompt

    # Dataset must be flat (no nested dicts) for HF datasets.
    from datasets import Dataset  # type: ignore

    rows: List[Dict[str, Any]] = []
    for t in raw_tasks:
        # Keep the dataset "flat" to avoid nested dict columns (which can break some collators/loggers).
        row: Dict[str, Any] = {}
        for k, v in dict(t).items():
            if isinstance(v, dict):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        row["prompt"] = make_prompt(t)
        rows.append(row)
    train_dataset = Dataset.from_list(rows)

    # Policy model + optional PEFT
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    peft_config = None
    if args.peft in ("lora", "qlora"):
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

    if args.peft == "qlora":
        device_map = None
        if torch.cuda.is_available() and not args.cpu:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device_map = {"": local_rank}

        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            quantization_config=qconf,
            torch_dtype=dtype,
            device_map=device_map,
        )
        # QLoRA best-practice: enable input grads + cast norms to fp32 (does NOT enable checkpointing here).
        try:
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except Exception:
            pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # In DDP, PyTorch's re-entrant checkpointing can trigger:
        # "Expected to mark a variable ready only once".
        # Prefer non-reentrant checkpointing when supported.
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    class _DDPStaticGraphCallback(TrainerCallback):
        """
        Workaround for some DDP + checkpointing edge cases:
          RuntimeError: Expected to mark a variable ready only once.
        """

        def on_train_begin(self, args, state, control, **kwargs):
            model = kwargs.get("model", None)
            try:
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
            except Exception:
                world_size = 1
            if world_size <= 1:
                return control
            if hasattr(model, "_set_static_graph"):
                try:
                    model._set_static_graph()
                except Exception:
                    pass
            return control

    class _ConsoleTrainLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            # HF Trainer typically logs `loss` plus LR/grad_norm; TRL may add extra keys.
            step = getattr(state, "global_step", None)
            loss = logs.get("loss", None)
            if loss is None:
                loss = logs.get("train_loss", None)

            try:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            except Exception:
                rank = 0
            if rank != 0:
                return control

            if args and getattr(args, "no_print_train_metrics", False):
                return control

            # Keep the console output compact and stable.
            pieces: List[str] = []
            if step is not None:
                pieces.append(f"step={int(step)}")
            if loss is not None:
                try:
                    pieces.append(f"loss={float(loss):.4f}")
                except Exception:
                    pieces.append(f"loss={loss}")
            if "learning_rate" in logs:
                try:
                    pieces.append(f"lr={float(logs['learning_rate']):.2e}")
                except Exception:
                    pass
            if "grad_norm" in logs:
                try:
                    pieces.append(f"grad_norm={float(logs['grad_norm']):.3f}")
                except Exception:
                    pass
            if "epoch" in logs:
                try:
                    pieces.append(f"epoch={float(logs['epoch']):.2f}")
                except Exception:
                    pass

            # If TRL exposes reward/KL logs, surface them too.
            for k in ("reward", "mean_reward", "kl", "mean_kl"):
                if k in logs:
                    try:
                        pieces.append(f"{k}={float(logs[k]):.4f}")
                    except Exception:
                        pass

            if pieces:
                print("[train] " + " ".join(pieces))
            return control

    # Reward function in TRL GRPO format.
    def reward_func(prompts: List[str], completions: List[str], completion_ids=None, **kwargs) -> List[float]:
        # `kwargs` are repeated per completion (B*G) by TRL.
        problems = kwargs.get("question_content") or kwargs.get("problem") or [""] * len(completions)
        starters = kwargs.get("starter_code") or [""] * len(completions)
        input_outputs = kwargs.get("input_output") or [None] * len(completions)
        ref_subgoals_col = kwargs.get("subgoals") or None
        ref_gap_col = kwargs.get("gap_analysis") or None

        rewards: List[float] = []
        for i, text in enumerate(completions):
            problem = str(problems[i] or "").strip()
            starter = str(starters[i] or "").strip()

            # Dense reward: subgoal alignment against dataset-provided subgoals if available.
            intended: List[str] = []
            if isinstance(ref_subgoals_col, list) and i < len(ref_subgoals_col) and isinstance(ref_subgoals_col[i], list):
                intended = [str(x) for x in ref_subgoals_col[i] if str(x).strip()]
            if not intended:
                intended = default_intended_subgoals(problem, max_n=args.n_subgoal_steps) or [problem] * args.n_subgoal_steps

            sg_block = _extract_tag(text, _TAG_SUBGOAL)
            sgs = [ln.strip() for ln in sg_block.splitlines() if ln.strip()] if sg_block else []
            sgs = sgs[: args.n_subgoal_steps]
            subgoal_score = 0.0
            if sgs:
                sub_sum = 0.0
                for j, sg in enumerate(sgs):
                    tgt = intended[j] if j < len(intended) else intended[-1]
                    sub_sum += scorer.score(problem=problem, generated_step=sg, intended_subgoal=str(tgt))
                subgoal_score = sub_sum / max(1, len(sgs))

            gap_text = _extract_tag(text, _TAG_GAP)
            gap_score = 0.0
            if gap_text and isinstance(ref_gap_col, list) and i < len(ref_gap_col) and ref_gap_col[i]:
                gap_score = scorer.score(problem=problem, generated_step=gap_text, intended_subgoal=str(ref_gap_col[i]))

            dense = args.dense_subgoal_weight * subgoal_score + args.dense_gap_weight * gap_score

            # Sparse terminal: run tests on generated code.
            code = _extract_code_fence(_extract_tag(text, _TAG_FINAL) or text)
            task_for_exec: Dict[str, Any] = {"question_content": problem, "starter_code": starter}
            if input_outputs[i] is not None:
                task_for_exec["input_output"] = input_outputs[i]
            tr = executor.run_tests(task=task_for_exec, solution_code=code)
            terminal = 1.0 if tr.passed else 0.0

            rewards.append(args.dense_weight * dense + args.terminal_weight * terminal)
        return rewards

    processing_class = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="no" if args.no_save_checkpoints else "steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=not args.bf16,
        max_prompt_length=args.max_prompt_tokens,
        max_completion_length=args.max_new_tokens,
        num_generations=num_generations,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        report_to=[] if args.log_with is None else [args.log_with],
        seed=args.seed,
    )
    # Often required with gradient checkpointing + DDP to avoid re-entrant backward edge cases.
    grpo_args.ddp_find_unused_parameters = False

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=processing_class,
        callbacks=[
            *([_DDPStaticGraphCallback()] if (args.ddp_static_graph or args.gradient_checkpointing) else []),
            _ConsoleTrainLogCallback(),
        ],
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if not args.no_save_final:
        trainer.save_model(args.output_dir)
        processing_class.save_pretrained(args.output_dir)
        (Path(args.output_dir) / "base_model.json").write_text(
            json.dumps({"base_model": args.model_name_or_path, "peft": args.peft}, indent=2),
            encoding="utf-8",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL fine-tuning (PPO/GRPO) for DeepSeek Prover on code tasks.")

    # Model / algo
    p.add_argument("--model-name-or-path", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    p.add_argument("--algo", type=str, choices=["ppo", "grpo"], default="grpo")
    p.add_argument("--output-dir", type=str, default="rl_deepseek_prover_out")
    p.add_argument("--cpu", action="store_true", help="Force CPU (debug only)")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 weights if supported")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from-checkpoint", type=str, default=None, help="HF Trainer checkpoint dir to resume from")
    p.add_argument("--no-print-train-metrics", action="store_true", help="Disable step/loss console logs during GRPO")
    p.add_argument("--no-save-checkpoints", action="store_true", help="Disable intermediate checkpoint saving during training")
    p.add_argument("--no-save-final", action="store_true", help="Do not save final model/tokenizer to --output-dir")
    p.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="(unused for LiveCodeBench offline loader; kept for compatibility)",
    )

    # Tasks
    p.add_argument(
        "--dataset",
        type=str,
        choices=["livecodebench", "bigcodebench", "jsonl"],
        default="livecodebench",
        help="Training dataset source",
    )
    p.add_argument(
        "--tasks-jsonl",
        type=str,
        default=None,
        help="JSONL path/glob/dir/comma-list (e.g. itssm_lcb_v*.jsonl). If omitted for --dataset jsonl, auto-load itssm_lcb_v*.jsonl in CWD.",
    )
    p.add_argument("--limit", type=int, default=50, help="Max tasks to load")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)

    # BigCodeBench integration
    p.add_argument("--bcb-split", type=str, choices=["instruct", "complete"], default="complete")
    p.add_argument("--bcb-subset", type=str, choices=["full", "hard"], default="hard")
    # LiveCodeBench integration
    p.add_argument("--lcb-versions", type=str, default="v1,v2,v3,v4,v5", help="Comma-separated version_tag list")
    p.add_argument("--lcb-include-private-tests", action="store_true", help="Include private tests in terminal reward (slower)")

    p.add_argument("--executor", type=str, choices=["livecodebench", "bigcodebench", "subprocess"], default="livecodebench")

    # Reward configuration
    p.add_argument("--n-subgoal-steps", type=int, default=4)
    p.add_argument("--dense-weight", type=float, default=0.3)
    p.add_argument("--terminal-weight", type=float, default=0.7)
    p.add_argument("--dense-subgoal-weight", type=float, default=1.0, help="Weight inside dense reward for subgoals")
    p.add_argument("--dense-gap-weight", type=float, default=1.0, help="Weight inside dense reward for gap analysis")
    p.add_argument("--subgoal-reward", type=str, choices=["embedding", "rm"], default="embedding")
    p.add_argument(
        "--subgoal-reward-model",
        type=str,
        default="intfloat/e5-small-v2",
        help="Embedding encoder or reward model path (must be available locally if offline)",
    )
    p.add_argument("--subgoal-reward-device", type=str, default="cpu", help="cpu|cuda[:i]")
    p.add_argument("--subgoal-reward-max-length", type=int, default=256)

    # Test sandbox limits (BigCodeBench)
    p.add_argument("--min-time-limit", type=float, default=1.0)
    p.add_argument("--gt-time-limit", type=float, default=2.0)
    p.add_argument("--max-as-limit", type=int, default=30 * 1024)
    p.add_argument("--max-data-limit", type=int, default=30 * 1024)
    p.add_argument("--max-stack-limit", type=int, default=10)

    # Generation
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)

    # Memory adaptation (recommended for 2xA6000)
    p.add_argument("--peft", choices=["none", "lora", "qlora"], default="qlora")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--ddp-static-graph", action="store_true", help="Call DDP._set_static_graph() at train start")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument("--lora-target-modules", type=str, default="all-linear")
    p.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb-4bit-use-double-quant", action="store_true")

    # GRPO training knobs (HF Trainer-style)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--num-generations", type=int, default=4, help="GRPO group size (G)")
    p.add_argument("--beta", type=float, default=0.02, help="GRPO KL weight (0 disables KL term)")

    # PPO hyperparams + KL control
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--rollout-batch-size", type=int, default=32, help="Number of env-steps per PPO update")
    p.add_argument("--mini-batch-size", type=int, default=8)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--cliprange-value", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.1)
    p.add_argument("--kl-coef", type=float, default=0.02, help="Initial KL coefficient")
    p.add_argument("--target-kl", type=float, default=0.1, help="Target KL for adaptive KL controller")
    p.add_argument("--disable-adaptive-kl", action="store_true")

    # GRPO-only
    p.add_argument("--grpo-group-size", type=int, default=8, help="(deprecated) use --num-generations")

    # Logging / saving
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-with", type=str, default=None, help="Optional TRL logger: wandb|tensorboard")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.algo == "ppo":
        train_ppo(args)
    else:
        train_grpo(args)


if __name__ == "__main__":
    main()
