#!/usr/bin/env python3
"""
Generate an RL training dataset by reproducing the ITSSM Robust Reflexion pipeline:
  1) Coder drafts code (Qwen prompt structure)
  2) Prover produces: contracts + subgoals + gap analysis (+ robustness checklist)
  3) Coder revises code using the prover analysis

This dataset is meant to be consumed by `train_deepseek_prover_rl.py --dataset jsonl`,
so it includes:
  - `subgoals`: list[str] extracted from prover analysis
  - `gap_analysis`: string extracted from prover analysis
  - `draft_code` / `revised_code`
  - `input_output` (LiveCodeBench tests; public tests by default)

Important
---------
LiveCodeBench HF dataset caching can be unreadable/writable in some sandboxes due to lockfiles.
This script uses the offline Arrow-cache loader from `train_deepseek_prover_rl.py` instead of
calling `datasets.load_dataset(version_tag=...)`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reduce TF/protobuf conflicts (this repo environment has TF/protobuf mismatches).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from fdg_approach_qwen3_inter_refined_reflex_robust import (  # type: ignore
    ITSSMApproachGenerator,
    OpenAIClient,
    VLLMClient,
)
from train_deepseek_prover_rl import LiveCodeBenchExecutor, _load_tasks_from_livecodebench


class DeepSeekReasonerClient:
    """
    OpenAI-compatible DeepSeek API client (reasoner).

    Docs: https://api-docs.deepseek.com/
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-reasoner",
        timeout_s: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.timeout_s = timeout_s
        # Compatibility with `VLLMClient` usage in `ITSSMApproachGenerator`.
        self.vllm_url = base_url

        try:
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: `openai`. Install with `pip install openai`.") from exc

        # DeepSeek implements an OpenAI-compatible endpoint.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def test_connection(self) -> bool:
        try:
            _ = self.generate("Say 'ok'.", max_tokens=4, temperature=0.0, retries=1)
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        system: Optional[str] = None,
        retries: int = 5,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "DeepSeek API key is empty. Set `DEEPSEEK_API_KEY` or pass `--deepseek-api-key` (leave blank in code)."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[BaseException] = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                )
                msg = resp.choices[0].message
                content = None
                reasoning = None
                try:
                    # openai>=1.x returns pydantic-like objects with attributes.
                    content = getattr(msg, "content", None)
                    reasoning = getattr(msg, "reasoning_content", None)
                except Exception:
                    pass
                if isinstance(msg, dict):
                    content = msg.get("content", content)
                    reasoning = msg.get("reasoning_content", reasoning)

                # DeepSeek reasoner may populate `reasoning_content` and leave `content` empty.
                text = (content or reasoning or "").strip()
                return text
            except Exception as exc:  # pragma: no cover
                last_err = exc
                time.sleep(min(10.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"DeepSeek request failed after {retries} retries: {last_err}")


def _normalize_openai_base_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "http://localhost:1234/v1"
    url = url.rstrip("/")
    # vLLM OpenAI-compatible API is typically served under `/v1`.
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text or "", flags=re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m else "").strip()


def _as_bullets(block: str) -> str:
    lines = [ln.strip() for ln in (block or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    if any(ln.startswith(("-", "*")) for ln in lines):
        return "\n".join(lines)
    return "\n".join([f"- {ln}" for ln in lines])


def _split_analysis_and_checklist(combined: str) -> Tuple[str, str]:
    text = (combined or "").replace("\r\n", "\n")
    if "<subgoal>" in text or "<gap_analysis>" in text or "<checklist>" in text:
        sub = _extract_tag(text, "subgoal")
        gap = _extract_tag(text, "gap_analysis")
        chk = _extract_tag(text, "checklist")
        analysis = "1. Preconditions and postconditions\n- (omitted)\n\n2. Key invariants / subgoals\n"
        analysis += (_as_bullets(sub) or "- (none)\n")
        analysis += "\n\n3. Concise gap analysis\n"
        analysis += (_as_bullets(gap) or "- (none)\n")
        return analysis.strip(), chk.strip()
    m = re.search(
        r"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?\s*Robustness\s+Checklist\s*(?:\*\*)?\s*:?\s*\n",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return text.strip(), ""
    left = text[: m.start()].strip()
    right = text[m.end() :].strip()
    return left, right


def _extract_section(text: str, start_pat: str, end_pat: Optional[str] = None) -> str:
    flags = re.IGNORECASE | re.DOTALL
    if end_pat is None:
        m = re.search(start_pat + r"(.*)$", text, flags)
        return (m.group(1).strip() if m else "").strip()
    m = re.search(start_pat + r"(.*?)" + end_pat, text, flags)
    return (m.group(1).strip() if m else "").strip()


def extract_subgoals_and_gap_analysis(type_analysis: str) -> Tuple[List[str], str]:
    """
    Best-effort extraction from prover output that was requested to include:
      1. Preconditions and postconditions
      2. Key invariants / subgoals
      3. Concise gap analysis

    Returns:
      subgoals: list[str]
      gap: str
    """

    # Normalize markdown a bit (DeepSeek often uses headings/bold instead of the numbered schema).
    ta = (type_analysis or "").replace("\r\n", "\n")
    ta_search = re.sub(r"[*_`]+", "", ta)

    # Try numbered sections first (original prompt format).
    subgoals_block = _extract_section(
        ta_search,
        start_pat=r"(?:^|\n)\s*(?:#+\s*)?2\s*[\.\)]\s*Key\s+invariants\s*/\s*subgoals\s*:?\s*\n",
        end_pat=r"(?:\n\s*3\s*[\.\)]\s*Concise\s+gap\s+analysis|\Z)",
    )
    gap_block = _extract_section(
        ta_search,
        start_pat=r"(?:^|\n)\s*(?:#+\s*)?3\s*[\.\)]\s*Concise\s+gap\s+analysis\s*:?\s*\n",
        end_pat=r"\Z",
    )

    # Fallback: look for loose headings (including markdown headings).
    if not subgoals_block:
        subgoals_block = _extract_section(
            ta_search,
            start_pat=r"(?:^|\n)\s*(?:#+\s*)?(Key\s+invariants\s*/\s*subgoals)\s*:?\s*\n",
            end_pat=r"(?:\n\s*(?:#+\s*)?(Concise\s+gap\s+analysis)|\Z)",
        )
    if not gap_block:
        gap_block = _extract_section(
            ta_search,
            start_pat=r"(?:^|\n)\s*(?:#+\s*)?(Concise\s+gap\s+analysis|Gap\s+analysis)\s*:?\s*\n",
            end_pat=r"\Z",
        )
    if not gap_block:
        m = re.search(r"(?:^|\n)\s*(No\s+gaps?.{0,2000})\s*$", ta_search, flags=re.IGNORECASE | re.DOTALL)
        if m:
            gap_block = (m.group(1) or "").strip()

    # If the analysis uses a "Subgoals:" subheading, focus on that portion.
    if subgoals_block:
        m = re.search(r"(?:^|\n)\s*(?:\*\*)?\s*Subgoals?\s*(?:\*\*)?\s*:?\s*\n(.*)$", subgoals_block, re.I | re.S)
        if m:
            subgoals_block = (m.group(1) or "").strip()

    # Normalize subgoals into a short list.
    lines = [ln.strip() for ln in subgoals_block.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", ln).strip()
        ln = re.sub(r"^\s*(?:SG\s*\d+|Subgoal\s*\d+)\s*:?\s*", "", ln, flags=re.IGNORECASE).strip()
        if ln:
            cleaned.append(ln)
    # If it's a paragraph, split into sentences.
    if len(cleaned) <= 1 and subgoals_block and len(subgoals_block) > 200:
        sents = re.split(r"(?<=[.!?])\s+", subgoals_block.strip())
        cleaned = [s.strip() for s in sents if len(s.strip()) >= 12][:8]

    # Deduplicate conservatively.
    out: List[str] = []
    seen = set()
    for sg in cleaned:
        key = sg.lower()
        if key not in seen:
            seen.add(key)
            out.append(sg)
    return out, gap_block.strip()


def _fallback_checklist_from_gap(gap: str) -> str:
    gap_lines = [ln.strip("-* \t") for ln in (gap or "").splitlines() if ln.strip()]
    out: List[str] = []
    for ln in gap_lines[:4]:
        if len(ln) >= 8:
            out.append(ln)
    if not out:
        out = [
            "Validate edge cases and boundary inputs.",
            "Verify output formatting and ordering requirements.",
            "Check algorithmic complexity against constraints.",
        ]
    return _as_bullets("\n".join(out))


def _existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("question_id") or json.loads(line).get("task_id")))
            except Exception:
                continue
    return ids


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate ITSSM-derived RL dataset from LiveCodeBench v1-v5.")
    p.add_argument("--output-jsonl", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="itssm_rl_dataset_artifacts")
    p.add_argument("--append", action="store_true", help="Append and skip existing ids in output file")

    # LiveCodeBench selection
    p.add_argument("--lcb-versions", type=str, default="v3")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--include-private-tests", action="store_true", help="Include private tests in `input_output`")

    # Prover backend for analysis/subgoals/gap analysis.
    p.add_argument("--prover-backend", choices=["deepseek", "vllm"], default="deepseek")
    p.add_argument("--prover-url", type=str, default=os.getenv("PROVER_BASE_URL", "http://localhost:5678/"))
    p.add_argument("--deepseek-api-key", type=str, default="", help="Leave blank in code; set via env DEEPSEEK_API_KEY")
    p.add_argument("--deepseek-base-url", type=str, default="https://api.deepseek.com")
    p.add_argument("--deepseek-model", type=str, default="deepseek-reasoner")

    # Coder backend (same prompt structure as ITSSM).
    p.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    p.add_argument(
        "--qwen-base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="vLLM OpenAI-compatible base URL (must include /v1); default matches the original script",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--reflections", type=int, default=1)

    # Optional labeling
    p.add_argument("--run-tests", action="store_true", help="Execute revised code on LCB tests and store pass/fail")
    p.add_argument("--test-timeout", type=int, default=6)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    skip_ids = _existing_task_ids(out_path) if args.append else set()
    mode = "a" if args.append else "w"

    # Match the original script: default to localhost:1234/v1 (OpenAI-compatible).
    os.environ["QWEN_VLLM_BASE_URL"] = _normalize_openai_base_url(args.qwen_base_url)

    if args.prover_backend == "deepseek":
        prover_client = DeepSeekReasonerClient(
            api_key=(args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")),
            base_url=args.deepseek_base_url,
            model_name=args.deepseek_model,
        )
    else:
        prover_client = VLLMClient(base_url=args.prover_url, model_name="deepseek-ai/DeepSeek-Prover-V2-7B")

    coder_client = OpenAIClient(
        api_key="EMPTY",
        model_name=args.qwen_model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    generator = ITSSMApproachGenerator(
        prover_client=prover_client,
        coder_client=coder_client,
        output_dir=str(artifacts_dir),
        reflections=args.reflections,
    )

    versions = [v.strip() for v in args.lcb_versions.split(",") if v.strip()]
    tasks = _load_tasks_from_livecodebench(
        versions=versions,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        include_private_tests=args.include_private_tests,
    )
    if not tasks:
        raise RuntimeError("No LiveCodeBench tasks loaded.")

    executor = LiveCodeBenchExecutor(timeout_s=int(args.test_timeout)) if args.run_tests else None

    written = 0
    checklist_missing_before_retry = 0
    checklist_missing_after_retry = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            task_id = task.get("question_id") or f"task_{i}"
            if task_id in skip_ids:
                continue

            print(f"[{written+1}/{len(tasks)}] {task_id}")
            try:
                draft = generator.stage1_initial(task)
                combined = generator.stage2_analysis_and_checklist(task, draft)
                revised = generator.stage3_revision(task, draft, combined)

                type_analysis, checklist = _split_analysis_and_checklist(combined)
                subgoals, gap = extract_subgoals_and_gap_analysis(type_analysis)
                if not checklist.strip():
                    checklist_missing_before_retry += 1
                    try:
                        checklist_retry = generator._prover_robustness_checklist(task, draft)  # type: ignore[attr-defined]
                        checklist = str(checklist_retry or "").strip()
                    except Exception:
                        checklist = ""
                if not checklist.strip():
                    checklist = _fallback_checklist_from_gap(gap)
                if not checklist.strip():
                    checklist_missing_after_retry += 1

                record: Dict[str, Any] = {
                    "question_id": task_id,
                    "lcb_version": task.get("lcb_version"),
                    "question_title": task.get("question_title", ""),
                    "question_content": task.get("question_content", ""),
                    "starter_code": task.get("starter_code", ""),
                    "input_output": task.get("input_output", ""),
                    "metadata": task.get("metadata", {}),
                    "draft_code": draft,
                    "type_analysis": type_analysis,
                    "robustness_checklist": checklist,
                    "subgoals": subgoals,
                    "gap_analysis": gap,
                    "revised_code": revised,
                }

                if executor is not None:
                    tr = executor.run_tests(task=task, solution_code=revised)
                    record["passed"] = bool(tr.passed)
                    record["test_status"] = tr.status
                    record["test_details"] = tr.details

                f.write(json.dumps(record) + "\n")
                written += 1
            except Exception as exc:
                err = {"question_id": task_id, "error": repr(exc)}
                f.write(json.dumps(err) + "\n")

    print(f"Done. Wrote {written} records to {out_path}")
    print(
        "[data] Checklist fill stats:"
        f" empty_before_retry={checklist_missing_before_retry}"
        f" empty_after_retry={checklist_missing_after_retry}"
    )


if __name__ == "__main__":
    main()
