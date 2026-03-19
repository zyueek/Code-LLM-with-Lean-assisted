#!/usr/bin/env python3
"""
ITSSM (Qwen3) + Robust Reflexion on BigCodeBench using a post-trained prover critic.

This is the BigCodeBench counterpart of:
  `fdg_approach_qwen3_inter_posttrained.py`

Key features (ported from the LiveCodeBench best run settings):
- Prover backend switch:
  - `vllm`: use a running prover server (OpenAI-compatible)
  - `hf`: load a local HuggingFace checkpoint / PEFT adapter (your RL-trained prover)
- Stage-specific decode controls:
  - `--draft-max-tokens`, `--revision-max-tokens`
  - `--draft-temperature`, `--revision-temperature`
- Search to reduce sampling churn:
  - `--draft-samples N`, `--revision-samples N`
  - selection is driven by optional self-test execution (see below)
- Reflexion loop: `--reflections K`

Outputs:
- Detailed ITSSM trace JSON in `--output-dir`
- BigCodeBench `samples.jsonl` in `--bcb-root` (sanitized)

Self-test note (important):
- If you enable `--self-test-public`, this script executes the BigCodeBench task tests locally
  via `bigcodebench.eval.untrusted_check` and uses pass/fail as a selection signal.
- This can substantially improve results but may be considered "test leakage" depending on the
  evaluation protocol you follow. Keep it disabled if you want a pure generation setting.

Example (best-run-aligned settings; adjust ranges):
  python fdg_approach_qwen3_posttrained_bigcodebench.py \
    --prover-backend hf \
    --prover-model-name-or-path /home/yueke/formalgen/lean_gen/rl_prover_critic_v3 \
    --prover-base-model-name-or-path deepseek-ai/DeepSeek-Prover-V2-7B \
    --require-checkpoint-best \
    --prover-format tags --stage2-mode combined \
    --qwen-base-url http://localhost:1234/v1 \
    --qwen-model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --coder-max-tokens 25000 --draft-max-tokens 12000 --revision-max-tokens 12000 \
    --temperature 0.6 --top-p 0.9 --seed 0 --revision-temperature 0.2 \
    --draft-samples 2 --revision-samples 2 --reflections 2 \
    --self-test-public --self-test-public-mode guard --no-trust-public-tests \
    --bcb-split complete --bcb-subset hard --start-index 0 --end-index 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "EMPTY")

# Avoid torch.compile / inductor subprocess compilation in restricted environments.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

BASE_DIR = Path(__file__).parent

# Import vLLMClient (same pattern as other scripts in this repo).
sys.path.insert(0, str(BASE_DIR.parent))
from vllm_client import VLLMClient  # noqa: E402

from fdg_approach_qwen3_inter_posttrained import (  # noqa: E402
    HFProverClient,
    PosttrainedITSSMApproachGenerator,
    VLLMCoderClient,
)


def _ensure_bigcodebench_importable() -> None:
    """Add local BigCodeBench repo to sys.path and verify imports."""

    bigcodebench_root = BASE_DIR / "bigcodebench"
    if bigcodebench_root.exists():
        sys.path.insert(0, str(bigcodebench_root))

    try:
        from bigcodebench.data import get_bigcodebench  # noqa: F401
        from bigcodebench.sanitize import sanitize  # noqa: F401
        from bigcodebench.eval import untrusted_check  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local BigCodeBench setup
        raise RuntimeError(
            "BigCodeBench repo is not importable. Make sure it is cloned at "
            "'lean_gen/bigcodebench' and installed with:\n"
            "  cd lean_gen/bigcodebench && pip install -e .\n"
        ) from exc


def _format_doc_struct(doc_struct_raw: str) -> str:
    """
    BigCodeBench stores `doc_struct` as a JSON-encoded string in the cached jsonl.
    Format it into a compact, readable block for prompting.
    """

    if not doc_struct_raw:
        return ""
    try:
        info = json.loads(doc_struct_raw)
    except Exception:
        return str(doc_struct_raw)

    if not isinstance(info, dict):
        return str(doc_struct_raw)

    lines: List[str] = []
    for key, value in info.items():
        if isinstance(value, list):
            body = "\n".join(f"  - {item}" for item in value)
        else:
            body = str(value)
        lines.append(f"{str(key).capitalize()}:\n{body}")
    return "\n".join(lines)


def _iter_bigcodebench_cached_rows(*, subset: str, version: str = "v0.1.4"):
    """
    Iterate BigCodeBench rows without requiring network access.

    BigCodeBench's `get_bigcodebench()` currently calls `datasets.load_dataset()` even when a
    cache jsonl exists. We prefer a direct read from:
      ~/.cache/bigcodebench/BigCodeBench[-Hard]-<version>.jsonl

    If BIGCODEBENCH_OVERRIDE_PATH is set, use it.
    """

    override = os.environ.get("BIGCODEBENCH_OVERRIDE_PATH")
    if override:
        path = Path(override).expanduser()
    else:
        cache_root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "bigcodebench"
        suffix = "-Hard" if subset == "hard" else ""
        path = cache_root / f"BigCodeBench{suffix}-{version}.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"BigCodeBench cache file not found: {path}. "
            "Either run `bigcodebench.evaluate` once to populate cache, or set BIGCODEBENCH_OVERRIDE_PATH."
        )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_bigcodebench_tasks(
    *,
    split: str = "complete",
    subset: str = "hard",
    question_source: str = "instruct",
    include_doc_struct: bool = True,
    include_libs: bool = True,
    include_tests: bool = False,
) -> List[Dict[str, Any]]:
    """Load BigCodeBench tasks and adapt them to ITSSM input format.

    `split` controls the *label* for the run and matches BigCodeBench CLI args.
    `question_source` controls what we feed into ITSSM as the natural-language spec:
      - instruct: use `instruct_prompt` (recommended for ITSSM synthesis)
      - complete: use `complete_prompt` (often includes code context + docstring and can be redundant)
    """

    _ensure_bigcodebench_importable()
    tasks: List[Dict[str, Any]] = []

    qs = str(question_source or "instruct").strip().lower()
    if qs not in {"instruct", "complete"}:
        raise ValueError("--bcb-question-source must be 'instruct' or 'complete'")

    for task in _iter_bigcodebench_cached_rows(subset=subset):
        task_id = task.get("task_id") or ""
        if not task_id:
            continue

        instruct_prompt = str(task.get("instruct_prompt") or "").strip()
        complete_prompt = str(task.get("complete_prompt") or "").strip()
        starter_code = str(task.get("code_prompt") or "").rstrip()
        entry_point = str(task.get("entry_point") or "").strip()
        libs = task.get("libs") or []
        if not isinstance(libs, list):
            libs = []
        doc_struct_text = _format_doc_struct(str(task.get("doc_struct") or "")) if include_doc_struct else ""

        base_spec = instruct_prompt if qs == "instruct" else complete_prompt
        header_lines = [
            f"You are solving a BigCodeBench-{subset.capitalize()} {split} task.",
            f"Implement `{entry_point}` so it satisfies the specification.",
            "The autograder prepends the starter code and calls this function directly.",
            "Do not read from stdin; do not print unless the spec explicitly requires it.",
            "Avoid CLI wrappers and `if __name__ == '__main__'` blocks.",
            "Keep the solution self-contained and deterministic.",
        ]
        if include_libs and libs:
            header_lines.append("Preferred/allowed libraries: " + ", ".join(map(str, libs)) + ". Avoid other imports.")
        if doc_struct_text:
            header_lines.append("Structured details:\n" + doc_struct_text)
        question = "\n".join(header_lines) + "\n\nSpecification (verbatim):\n" + base_spec

        adapted: Dict[str, Any] = {
            # ITSSM uses question_id; BigCodeBench evaluator expects task_id in samples.jsonl.
            "question_id": task_id,
            "task_id": task_id,
            "question_content": question,
            "starter_code": starter_code,
            # Keep both prompts for testing/eval alignment.
            "instruct_prompt": instruct_prompt,
            "complete_prompt": complete_prompt,
            "entry_point": entry_point,
            "libs": libs,
            "doc_struct_text": doc_struct_text,
        }
        if include_tests:
            adapted["test"] = str(task.get("test") or "")
        tasks.append(adapted)

    return tasks


def _sanitize_bigcodebench_solution(*, code: str, entry_point: str) -> str:
    if not code.strip():
        return ""
    try:
        _ensure_bigcodebench_importable()
        from bigcodebench.sanitize import sanitize

        return sanitize(code, entry_point or None)
    except Exception:
        return code


@dataclass(frozen=True)
class _PublicTestResult:
    score: float
    status: str
    details: Dict[str, str]
    summary: str


class _BigCodeBenchSelfTestExecutor:
    """Self-test runner used for candidate selection (pass=1.0 else partial score)."""

    def __init__(
        self,
        *,
        timeout_s: int,
        calibrated: bool,
        max_as_limit: int,
        max_data_limit: int,
        max_stack_limit: int,
    ):
        self.timeout_s = max(1.0, float(timeout_s))
        self.calibrated = bool(calibrated)
        self.max_as_limit = int(max_as_limit)
        self.max_data_limit = int(max_data_limit)
        self.max_stack_limit = int(max_stack_limit)
        _ensure_bigcodebench_importable()
        from bigcodebench.eval import untrusted_check

        self._untrusted_check = untrusted_check

    @staticmethod
    def _extract_failure_line(trace: str, *, max_chars: int = 240) -> str:
        lines = [ln.strip() for ln in str(trace).splitlines() if ln.strip()]
        msg = lines[-1] if lines else ""
        if len(msg) > int(max_chars):
            msg = msg[: max(0, int(max_chars) - 3)] + "..."
        return msg

    def run(self, *, task: Dict[str, Any], solution_code: str) -> _PublicTestResult:
        entry_point = str(task.get("entry_point") or "").strip()
        sanitized = _sanitize_bigcodebench_solution(code=solution_code or "", entry_point=entry_point)
        test_code = str(task.get("test") or "").strip()
        if not test_code or not entry_point:
            return _PublicTestResult(score=0.0, status="missing", details={}, summary="(missing tests or entry_point)")

        full = sanitized
        if self.calibrated:
            code_prompt = str(task.get("starter_code") or "").rstrip()
            # Match BigCodeBench evaluate.py: `code_prompt + "\n    pass\n" + solution`.
            full = code_prompt + "\n    pass\n" + sanitized.lstrip()

        try:
            status, details = self._untrusted_check(
                full,
                test_code,
                entry_point,
                self.max_as_limit,
                self.max_data_limit,
                self.max_stack_limit,
                1.0,
                self.timeout_s,
            )
        except Exception:
            return _PublicTestResult(score=0.0, status="error", details={}, summary="(self-test error)")

        status_str = str(status).strip().lower()
        details_dict: Dict[str, str] = {}
        try:
            if isinstance(details, dict):
                details_dict = {str(k): str(v) for k, v in details.items()}
        except Exception:
            details_dict = {}

        if status_str == "pass":
            if details_dict:
                status_str = "fail"
            else:
                return _PublicTestResult(score=1.0, status="pass", details={}, summary="PASS")

        if status_str != "fail":
            return _PublicTestResult(score=0.0, status=status_str, details=details_dict, summary=status_str.upper())

        # Partial score for selection: prefer candidates that fail fewer tests.
        n_fail = len(details_dict) if details_dict else 99
        score = 1.0 / float(1 + int(n_fail))
        summary_lines: List[str] = []
        if details_dict:
            for name, trace in list(details_dict.items())[:3]:
                msg = self._extract_failure_line(trace)
                if not msg:
                    continue
                summary_lines.append(f"- {name}: {msg}")
        summary = f"FAIL ({len(details_dict)} failing tests)\n" + ("\n".join(summary_lines) if summary_lines else "")
        return _PublicTestResult(score=float(score), status="fail", details=details_dict, summary=summary.strip())

    def score(self, *, task: Dict[str, Any], solution_code: str) -> float:
        return float(self.run(task=task, solution_code=solution_code).score)


def _prefer_candidate_on_tie(*, cand: str, cur: str) -> bool:
    """Heuristic tie-breaker when self-test scores are equal."""

    cand_len = len(cand or "")
    cur_len = len(cur or "")
    if cand_len != cur_len:
        return cand_len < cur_len
    return False


class PosttrainedBigCodeBenchGenerator(PosttrainedITSSMApproachGenerator):
    """Posttrained ITSSM generator wired to BigCodeBench self-tests (optional)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Disable the LiveCodeBench self-test inside the base class; we replace it.
        self_test_public = bool(kwargs.pop("self_test_public", False))
        self_test_timeout_s = int(kwargs.get("self_test_timeout_s", 6))
        bcb_calibrated = bool(kwargs.pop("bcb_calibrated", True))
        bcb_max_as_limit = int(kwargs.pop("bcb_max_as_limit", 30 * 1024))
        bcb_max_data_limit = int(kwargs.pop("bcb_max_data_limit", 30 * 1024))
        bcb_max_stack_limit = int(kwargs.pop("bcb_max_stack_limit", 10))
        super().__init__(*args, self_test_public=False, **kwargs)
        if self_test_public:
            self._public_tester = _BigCodeBenchSelfTestExecutor(
                timeout_s=self_test_timeout_s,
                calibrated=bcb_calibrated,
                max_as_limit=bcb_max_as_limit,
                max_data_limit=bcb_max_data_limit,
                max_stack_limit=bcb_max_stack_limit,
            )
        else:
            self._public_tester = None

    def generate_one(self, problem: Dict[str, Any], index: int):
        # BigCodeBench-tuned variant of PosttrainedITSSMApproachGenerator.generate_one():
        # - uses a partial self-test score (based on number of failing tests) to reduce tie churn
        # - injects a short public-test failure summary into the revision prompt signal

        from fdg_approach_qwen3_inter_refined_reflex_robust import ITSSMResult

        task_id = self.get_task_id(problem, index)
        print(f"\n🎯 ITSSM Robust Reflex for {task_id}")

        def _hint_from_score(score: Optional[float]) -> Optional[bool]:
            if score is None:
                return None
            if score < 0.999999:
                return True
            return False if self._trust_public_tests else None

        def _score_public(code: str) -> tuple[Optional[float], str]:
            if self._public_tester is None:
                return None, ""
            try:
                if hasattr(self._public_tester, "run"):
                    tr = self._public_tester.run(task=problem, solution_code=code)  # type: ignore[attr-defined]
                    return float(tr.score), str(tr.summary or "")
                return float(self._public_tester.score(task=problem, solution_code=code)), ""
            except Exception:
                return None, ""

        best_code = ""
        best_score: float = -1.0

        def _update_best(code: str, score: Optional[float], summary: str) -> None:
            nonlocal best_code, best_score
            if self._public_tester is None or self._self_test_public_mode == "hint" or score is None:
                return
            s = float(score)
            if s > best_score:
                best_score = s
                best_code = code
                return
            if s == best_score:
                if _prefer_candidate_on_tie(cand=code, cur=best_code):
                    best_code = code
                    return
                if self._self_test_public_mode == "guard":
                    best_code = code

        base_seed = None
        try:
            if getattr(self.coder, "seed", None) is not None:
                base_seed = int(getattr(self.coder, "seed"))
        except Exception:
            base_seed = None

        # Stage 1: sample multiple drafts (selected by public tests when enabled).
        if self._draft_samples > 1 and (self._public_tester is None or self._self_test_public_mode == "hint"):
            print(
                f"⚠️ --draft-samples={self._draft_samples} requested but public-test selection is unavailable "
                f"(self_test_public={bool(self._public_tester is not None)} mode={self._self_test_public_mode}). "
                "Falling back to 1 draft."
            )
        actual_drafts = (
            self._draft_samples if (self._public_tester is not None and self._self_test_public_mode != "hint") else 1
        )

        draft = ""
        draft_score: Optional[float] = None
        draft_summary = ""
        for di in range(int(actual_drafts)):
            seed_i = (base_seed + di) if base_seed is not None else None
            cand = self.stage1_initial(problem, seed_override=seed_i)
            cand_score, cand_sum = _score_public(cand)
            if not draft:
                draft, draft_score, draft_summary = cand, cand_score, cand_sum
            else:
                if cand_score is not None and draft_score is not None:
                    better = float(cand_score) > float(draft_score)
                    tie = float(cand_score) == float(draft_score)
                    if better or (
                        tie
                        and (
                            _prefer_candidate_on_tie(cand=cand, cur=draft)
                            or self._self_test_public_mode == "guard"
                        )
                    ):
                        draft, draft_score, draft_summary = cand, cand_score, cand_sum
            _update_best(cand, cand_score, cand_sum)

        self._redteam_hint_fail = _hint_from_score(draft_score)
        combined = self.stage2_analysis_and_checklist(problem, draft)
        if draft_summary and (draft_score is not None and float(draft_score) < 0.999999):
            combined = combined.strip() + "\n\nPublic test signal:\n" + draft_summary.strip()

        # Stage 3: sample multiple revisions per critique (selected by public tests when enabled).
        if self._revision_samples > 1 and (self._public_tester is None or self._self_test_public_mode == "hint"):
            print(
                f"⚠️ --revision-samples={self._revision_samples} requested but public-test selection is unavailable "
                f"(self_test_public={bool(self._public_tester is not None)} mode={self._self_test_public_mode}). "
                "Falling back to 1 revision."
            )
        actual_revs = (
            self._revision_samples if (self._public_tester is not None and self._self_test_public_mode != "hint") else 1
        )

        revised = ""
        revised_score: Optional[float] = None
        revised_summary = ""
        for ri in range(int(actual_revs)):
            seed_i = (base_seed + 1000 + ri) if base_seed is not None else None
            cand = self.stage3_revision(problem, draft, combined, seed_override=seed_i)
            cand_score, cand_sum = _score_public(cand)
            if not revised:
                revised, revised_score, revised_summary = cand, cand_score, cand_sum
            else:
                if cand_score is not None and revised_score is not None:
                    better = float(cand_score) > float(revised_score)
                    tie = float(cand_score) == float(revised_score)
                    if better or (
                        tie
                        and (
                            _prefer_candidate_on_tie(cand=cand, cur=revised)
                            or self._self_test_public_mode == "guard"
                        )
                    ):
                        revised, revised_score, revised_summary = cand, cand_score, cand_sum
            _update_best(cand, cand_score, cand_sum)

        if revised_score is not None:
            self._redteam_hint_fail = _hint_from_score(revised_score)

        reflections_to_run = int(self.reflections)
        if (
            self._reflections_when_public_perfect
            and self._public_tester is not None
            and self._trust_public_tests
            and best_score >= 0.999999
        ):
            reflections_to_run = 0

        for r in range(reflections_to_run):
            print(f"♻️ Reflexion {r+1}/{self.reflections}")
            fb = self._prover_feedback(problem, revised)
            combined2 = combined + "\n\nReflexion Feedback:\n" + fb.strip()
            if revised_summary and (revised_score is not None and float(revised_score) < 0.999999):
                combined2 = combined2.strip() + "\n\nPublic test signal:\n" + revised_summary.strip()

            new_revised = ""
            new_score: Optional[float] = None
            new_summary = ""
            for ri in range(int(actual_revs)):
                seed_i = (base_seed + 2000 + r * 50 + ri) if base_seed is not None else None
                cand = self.stage3_revision(problem, revised, combined2, seed_override=seed_i)
                cand_score, cand_sum = _score_public(cand)
                if not new_revised:
                    new_revised, new_score, new_summary = cand, cand_score, cand_sum
                else:
                    if cand_score is not None and new_score is not None:
                        better = float(cand_score) > float(new_score)
                        tie = float(cand_score) == float(new_score)
                        if better or (
                            tie
                            and (
                                _prefer_candidate_on_tie(cand=cand, cur=new_revised)
                                or self._self_test_public_mode == "guard"
                            )
                        ):
                            new_revised, new_score, new_summary = cand, cand_score, cand_sum
                _update_best(cand, cand_score, cand_sum)

            if new_revised.strip() == revised.strip():
                print("🔁 No effective change; stop.")
                combined = combined2
                break
            revised, revised_score, revised_summary = new_revised, new_score, new_summary
            combined = combined2

        # If enabled, pick the best code by public tests.
        if self._public_tester is not None and best_score >= 0.0 and self._self_test_public_mode != "hint":
            revised = best_code or revised
        self._redteam_hint_fail = None

        return ITSSMResult(
            task_id=task_id,
            problem_description=problem.get("question_content", ""),
            initial_proof_term=draft,
            type_analysis=combined,
            revised_python=revised,
            full_function=revised,
            reasoning_chain=f"draft[{actual_drafts}]→analysis+checklist→revise[{actual_revs}]→reflex x{reflections_to_run}",
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ITSSM (posttrained prover + robust reflexion) on BigCodeBench (samples.jsonl output)."
    )

    # Prover backend
    p.add_argument(
        "--prover-backend",
        type=str,
        choices=["vllm", "hf"],
        default="hf",
        help="Prover backend: vllm server or local HuggingFace model/adapter.",
    )
    p.add_argument(
        "--prover-url",
        type=str,
        default=os.getenv("PROVER_BASE_URL", "http://localhost:5678/"),
        help="(vllm) Prover OpenAI-compatible base URL.",
    )
    p.add_argument(
        "--prover-model",
        type=str,
        default=os.getenv("PROVER_MODEL", "deepseek-ai/DeepSeek-Prover-V2-7B"),
        help="(vllm) Served model name/id to request from the prover server.",
    )
    p.add_argument("--prover-model-name-or-path", type=str, default=None, help="(hf) Prover model path or adapter dir.")
    p.add_argument("--prover-base-model-name-or-path", type=str, default=None, help="(hf, adapter only) Base model id/path.")
    p.add_argument("--prover-device", type=str, default="auto", help="(hf) auto|cpu|cuda|cuda:0|cuda:1")
    p.add_argument("--prover-load-in-4bit", action="store_true", help="(hf) Load base model in 4-bit (CUDA only).")
    p.add_argument("--prover-bf16", action="store_true", help="(hf) Use bf16 for non-4bit inference (if supported).")
    p.add_argument("--prover-merge-lora", action="store_true", help="(hf, adapter) Merge LoRA into base model for inference.")
    p.add_argument(
        "--prover-format",
        type=str,
        choices=["auto", "original", "tags"],
        default="auto",
        help="How to prompt/parse prover outputs. Use `tags` for GRPO-trained prover format.",
    )
    p.add_argument(
        "--stage2-mode",
        type=str,
        choices=["auto", "split", "combined"],
        default="auto",
        help="Stage-2 mode. `combined` matches rl_training_prover_process.py (subgoal+gap+checklist in one call).",
    )
    p.add_argument(
        "--require-checkpoint-best",
        action="store_true",
        help="(hf) If --prover-model-name-or-path is a run dir, require and use its checkpoint-best subdir.",
    )

    # BigCodeBench selection
    p.add_argument("--bcb-split", type=str, default="complete", choices=["instruct", "complete"])
    p.add_argument("--bcb-subset", type=str, default="hard", choices=["full", "hard"])
    p.add_argument("--bcb-root", type=str, default="bcb_results", help="Directory to write BigCodeBench samples.jsonl.")
    p.add_argument(
        "--bcb-question-source",
        type=str,
        default="instruct",
        choices=["instruct", "complete"],
        help="What to feed as the natural-language spec into ITSSM (recommended: instruct).",
    )
    p.add_argument(
        "--bcb-include-doc-struct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include BigCodeBench doc_struct field (formatted) in the prompt.",
    )
    p.add_argument(
        "--bcb-include-libs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include BigCodeBench libs hint in the prompt.",
    )
    p.add_argument(
        "--bcb-calibrated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match BigCodeBench evaluate.py `calibrated=True` wrapping during self-test selection.",
    )
    p.add_argument("--bcb-max-as-limit", type=int, default=30 * 1024)
    p.add_argument("--bcb-max-data-limit", type=int, default=30 * 1024)
    p.add_argument("--bcb-max-stack-limit", type=int, default=10)

    # Output / range
    p.add_argument("--output-dir", type=str, default="./itssm_approach_results", help="Directory to store detailed ITSSM traces.")
    p.add_argument("--run-timestamp", type=int, default=None, help="Fixed timestamp for reproducible output filenames.")
    p.add_argument("--test-connections", action="store_true", help="Only test endpoints/model loading and exit.")
    p.add_argument("--all", action="store_true", help="Ignore --max-problems and process the entire selected range.")
    p.add_argument("--max-problems", type=int, default=None, help="Max tasks to process (applied after slicing).")
    p.add_argument("--start-index", type=int, default=0, help="Start index (inclusive).")
    p.add_argument("--end-index", type=int, default=None, help="End index (exclusive).")

    # Coder config
    p.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    p.add_argument("--qwen-base-url", type=str, default=None, help="Coder vLLM base URL (default http://localhost:1234/v1).")
    p.add_argument("--coder-max-tokens", type=int, default=25000, help="Max tokens for coder generation (stage-1/stage-3).")
    p.add_argument("--coder-context-margin-tokens", type=int, default=256, help="Reserved headroom when recovering from context overflow.")
    p.add_argument("--coder-min-completion-tokens", type=int, default=256, help="Fail fast if overflow leaves fewer completion tokens.")
    p.add_argument("--seed", type=int, default=0, help="Optional vLLM seed for sampling (reproducibility).")

    # Decoding knobs
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    p.add_argument("--top-k", dest="top_k", type=int, default=20)
    p.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.05)
    p.add_argument("--draft-temperature", type=float, default=None, help="Optional override temperature for stage-1 draft generation.")
    p.add_argument("--revision-temperature", type=float, default=0.2, help="Optional override temperature for stage-3 revision generation.")
    p.add_argument("--draft-max-tokens", type=int, default=12000, help="Per-draft max tokens (caps --coder-max-tokens).")
    p.add_argument("--revision-max-tokens", type=int, default=12000, help="Per-revision max tokens (caps --coder-max-tokens).")

    # Search / reflexion knobs
    p.add_argument("--draft-samples", type=int, default=2, help="Number of draft samples to consider (requires self-test selection).")
    p.add_argument("--revision-samples", type=int, default=2, help="Number of revision samples to consider (requires self-test selection).")
    p.add_argument("--reflections", type=int, default=2, help="Number of reflexion iterations (>=0).")
    p.add_argument(
        "--reflections-when-public-perfect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true and tests are trusted, skip reflections when best self-test score is perfect.",
    )
    p.add_argument(
        "--prover-redteam-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry prover critique/feedback with a red-team prompt when output is overconfident or too thin.",
    )
    p.add_argument("--prover-redteam-max-retries", type=int, default=1, help="Max red-team retries per prover call.")

    # Self-test selection controls
    p.add_argument(
        "--self-test-public",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run BigCodeBench tests locally for selection among candidates (see leakage note in docstring).",
    )
    p.add_argument("--self-test-timeout-s", type=int, default=6, help="Approx per-task time limit used by BigCodeBench untrusted_check.")
    p.add_argument(
        "--trust-public-tests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, treat a perfect self-test score as a strong correctness hint (can reduce red-team pressure).",
    )
    p.add_argument(
        "--self-test-public-mode",
        type=str,
        choices=["select", "guard", "hint"],
        default="guard",
        help="Selection mode: select highest score; guard prefers later on ties; hint only (no selection).",
    )

    return p


def _build_prover(args: argparse.Namespace):
    if args.prover_backend == "vllm":
        return VLLMClient(base_url=args.prover_url, model_name=args.prover_model)

    if not args.prover_model_name_or_path:
        raise SystemExit("--prover-backend hf requires --prover-model-name-or-path")
    path = Path(args.prover_model_name_or_path)
    if args.require_checkpoint_best:
        cand = path / "checkpoint-best"
        if cand.exists() and cand.is_dir():
            path = cand
        else:
            raise SystemExit(f"--require-checkpoint-best set but not found: {cand}")
    return HFProverClient(
        model_name_or_path=str(path),
        base_model_name_or_path=args.prover_base_model_name_or_path,
        device=args.prover_device,
        load_in_4bit=bool(args.prover_load_in_4bit),
        bf16=bool(args.prover_bf16),
        merge_lora=bool(args.prover_merge_lora),
    )


def _build_coder(args: argparse.Namespace) -> VLLMCoderClient:
    qwen_base_url = args.qwen_base_url or os.getenv("QWEN_VLLM_BASE_URL", "http://localhost:1234/v1")
    return VLLMCoderClient(
        api_key="EMPTY",
        model_name=args.qwen_model,
        base_url=qwen_base_url,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        seed=(int(args.seed) if args.seed is not None else None),
        context_margin_tokens=int(args.coder_context_margin_tokens),
        min_completion_tokens=int(args.coder_min_completion_tokens),
    )


def run(args: argparse.Namespace) -> Path:
    # Build clients (prover + coder)
    prover = _build_prover(args)
    coder = _build_coder(args)

    if args.test_connections:
        print("[test] Testing connections...")
        prover_ok = bool(prover.test_connection())
        coder_ok = bool(coder.test_connection())
        if prover_ok and coder_ok:
            print("[test] OK")
            raise SystemExit(0)
        raise SystemExit("[test] Failed")

    generator = PosttrainedBigCodeBenchGenerator(
        prover,
        coder,
        args.output_dir,
        reflections=int(args.reflections),
        coder_max_tokens=int(args.coder_max_tokens),
        prover_format=str(args.prover_format),
        stage2_mode=str(args.stage2_mode),
        prover_redteam_retry=bool(args.prover_redteam_retry),
        prover_redteam_max_retries=int(args.prover_redteam_max_retries),
        draft_temperature=args.draft_temperature,
        revision_temperature=args.revision_temperature,
        draft_max_tokens=int(args.draft_max_tokens) if args.draft_max_tokens is not None else None,
        revision_max_tokens=int(args.revision_max_tokens) if args.revision_max_tokens is not None else None,
        draft_samples=int(args.draft_samples),
        revision_samples=int(args.revision_samples),
        reflections_when_public_perfect=bool(args.reflections_when_public_perfect),
        self_test_public=bool(args.self_test_public),
        self_test_timeout_s=int(args.self_test_timeout_s),
        trust_public_tests=bool(args.trust_public_tests),
        self_test_public_mode=str(args.self_test_public_mode),
        bcb_calibrated=bool(args.bcb_calibrated),
        bcb_max_as_limit=int(args.bcb_max_as_limit),
        bcb_max_data_limit=int(args.bcb_max_data_limit),
        bcb_max_stack_limit=int(args.bcb_max_stack_limit),
    )

    tasks = load_bigcodebench_tasks(
        split=str(args.bcb_split),
        subset=str(args.bcb_subset),
        question_source=str(args.bcb_question_source),
        include_doc_struct=bool(args.bcb_include_doc_struct),
        include_libs=bool(args.bcb_include_libs),
        include_tests=bool(args.self_test_public),
    )

    total = len(tasks)
    start_index = int(args.start_index)
    end_index = int(args.end_index) if args.end_index is not None else total
    if start_index < 0 or start_index >= total:
        raise ValueError(f"start_index ({start_index}) out of range [0, {total})")
    if end_index < start_index or end_index > total:
        raise ValueError(f"end_index ({end_index}) must be >= start_index and <= {total}")

    selected = tasks[start_index:end_index]
    if not args.all and args.max_problems is not None and int(args.max_problems) < len(selected):
        selected = selected[: int(args.max_problems)]

    results = []
    start_t = time.time()
    for offset, problem in enumerate(selected):
        idx = start_index + offset
        task_id = str(problem.get("question_id") or problem.get("task_id") or f"task_{idx}")
        print(f"\n--- Task {idx} ({offset+1}/{len(selected)}): {task_id} ---")
        try:
            res = generator.generate_one(problem, idx)
            results.append(res)
            print(f"[ok] {task_id}: revised_len={len(res.revised_python)}")
        except Exception as exc:
            print(f"[err] {task_id}: {exc}")
            from fdg_approach_qwen3_inter_refined_reflex_robust import ITSSMResult

            results.append(
                ITSSMResult(
                    task_id=task_id,
                    problem_description=problem.get("question_content", ""),
                    initial_proof_term=f"# Error: {exc}",
                    type_analysis=f"# Error: {exc}",
                    revised_python=f"# Error: {exc}",
                    full_function=f"# Error: {exc}",
                    reasoning_chain=f"Error: {exc}",
                )
            )

    end_t = time.time()
    run_ts = int(args.run_timestamp) if args.run_timestamp is not None else int(time.time())
    if run_ts <= 0:
        raise ValueError(f"run_timestamp must be positive when provided; got {run_ts}")

    # Write detailed trace JSON (ITSSM-style).
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"itssm_robust_reflex_bigcodebench_results_{run_ts}.json"
    trace = {
        "generation_info": {
            "approach": "ITSSM_robust_reflex_bigcodebench_posttrained",
            "benchmark": "BigCodeBench",
            "split": str(args.bcb_split),
            "subset": str(args.bcb_subset),
            "num_problems": len(results),
            "generation_time_seconds": float(end_t - start_t),
            "prover_model": getattr(prover, "vllm_url", getattr(prover, "model_name", "unknown")),
            "coder_model": getattr(coder, "model_name", "unknown"),
            "reflections": int(args.reflections),
            "draft_samples": int(args.draft_samples),
            "revision_samples": int(args.revision_samples),
            "self_test_public": bool(args.self_test_public),
        },
        "results": [
            {
                "task_id": r.task_id,
                "problem_description": r.problem_description,
                "initial_proof_term": r.initial_proof_term,
                "type_analysis": r.type_analysis,
                "revised_python": r.revised_python,
                "full_function": r.full_function,
                "reasoning_chain": r.reasoning_chain,
            }
            for r in results
        ],
    }
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"[write] Trace: {trace_path}")

    # Write BigCodeBench samples.jsonl
    bcb_root = Path(args.bcb_root)
    bcb_root.mkdir(parents=True, exist_ok=True)
    model_tag = str(args.qwen_model).replace("/", "--")
    samples_path = bcb_root / f"{model_tag}--bigcodebench-{args.bcb_subset}-{args.bcb_split}--itssm-posttrained-{run_ts}.jsonl"

    _ensure_bigcodebench_importable()
    with samples_path.open("w", encoding="utf-8") as out:
        for prob, res in zip(selected, results):
            task_id = str(prob.get("task_id") or prob.get("question_id") or res.task_id)
            entry_point = str(prob.get("entry_point") or "")
            raw_code = res.revised_python or ""
            if raw_code.strip().startswith("# Error") or not raw_code.strip():
                solution = raw_code.strip() or "# Error: empty generation"
            else:
                solution = _sanitize_bigcodebench_solution(code=raw_code, entry_point=entry_point)

            rec = {
                "task_id": task_id,
                "solution": solution,
                "raw_solution": raw_code,
            }
            out.write(json.dumps(rec) + "\n")

    print(f"[write] BigCodeBench samples: {samples_path}")
    print("[next] Evaluate (local):")
    print(
        "  bigcodebench.evaluate --execution local "
        f"--split {args.bcb_split} --subset {args.bcb_subset} --samples {samples_path}"
    )

    return samples_path


def main() -> None:
    p = _build_arg_parser()
    args = p.parse_args()
    if len(sys.argv) == 1:
        p.print_help()
        return
    run(args)


if __name__ == "__main__":
    main()
