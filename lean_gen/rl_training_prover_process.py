#!/usr/bin/env python3
"""
RL fine-tuning for DeepSeek Prover in the *ITSSM prover/critic role* (process-aligned).

Why this script exists
----------------------
`train_deepseek_prover_rl.py` trains a prover to emit tags + final code, but in the ITSSM pipeline
the prover is used as a *critic* that conditions a *coder*:

  Stage 1: coder drafts code
  Stage 2: prover produces analysis/checklist (NO code)
  Stage 3: coder revises using prover text
  Stage 4: optional reflexion feedback + another revision

This script trains the prover with GRPO where the *action* is the Stage-2 critic output:

  <subgoal>...</subgoal>
  <gap_analysis>...</gap_analysis>
  <checklist>...</checklist>

Rewards
-------
- Dense reward: alignment between generated (subgoals/gap/checklist) and ITSSM dataset targets
  (embedding cosine similarity by default).
- Sparse terminal reward: run the *coder revision* using the generated critic output and then
  run LiveCodeBench public tests on the revised code (pass=1, fail=0).

Datasets
--------
Loads JSONL files matching `itssm_lcb_v*.jsonl` by default (auto-detect in CWD), or via --tasks-jsonl.

Evaluation alignment
--------------------
This script is aligned to `fdg_approach_qwen3_inter_posttrained.py` when run with `--prover-format tags`
and stage2 "combined" mode (subgoals+gap+checklist in one call).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import hashlib
import shutil
import sys
import time
import logging
import shlex
import difflib
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Reduce TF/Flax side effects.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# Help mitigate CUDA allocator fragmentation (user can override).
# New name (PyTorch 2.6+): PYTORCH_ALLOC_CONF. Keep legacy name for older versions.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import TrainerCallback

from train_deepseek_prover_rl import (
    EmbeddingCosineScorer,
    FrozenRewardModelScorer,
    LiveCodeBenchExecutor,
    SubgoalAlignmentScorer,
)


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


def _resolve_api_key(explicit: str = "") -> str:
    """
    Resolve API key from explicit arg, common env vars, or optional key files.
    """
    candidates = [
        explicit,
        os.getenv("QUALITY_LLM_API_KEY", ""),
        os.getenv("AMPLIFY_API_KEY", ""),
        os.getenv("AMPLIFY_API_TOKEN", ""),
        os.getenv("AMPLIFY_TOKEN", ""),
        os.getenv("AMP_API_KEY", ""),
        os.getenv("AMP_TOKEN", ""),
        os.getenv("amplify_api_key", ""),
    ]
    for value in candidates:
        key = str(value or "").strip()
        if key:
            return key

    key_file_candidates = [
        os.getenv("QUALITY_LLM_API_KEY_FILE", ""),
        os.getenv("AMPLIFY_API_KEY_FILE", ""),
        str(Path.cwd() / ".amplify_api_key"),
        str(Path.home() / ".amplify_api_key"),
    ]
    for fp in key_file_candidates:
        path = Path(str(fp or "").strip())
        if not str(path):
            continue
        try:
            if path.exists() and path.is_file():
                key = path.read_text(encoding="utf-8").strip()
                if key:
                    return key
        except Exception:
            continue
    return ""


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
    def _redact_cli(argv: List[str]) -> List[str]:
        sensitive_flags = {"--quality-llm-api-key"}
        out: List[str] = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            if any(tok.startswith(f"{flag}=") for flag in sensitive_flags):
                out.append(tok.split("=", 1)[0] + "=***")
                i += 1
                continue
            if tok in sensitive_flags:
                out.append(tok)
                if i + 1 < len(argv):
                    out.append("***")
                    i += 2
                else:
                    i += 1
                continue
            out.append(tok)
            i += 1
        return out

    print(f"[log] Writing training log to {log_path}")
    print("[log] Command:", " ".join(_redact_cli(list(sys.argv))))
    return f


def _rank0() -> bool:
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except Exception:
        rank = 0
    return rank == 0


class _JSONLMetricsCallback(TrainerCallback):
    def __init__(self, fh):
        self._fh = fh

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self._fh is None:
            return control
        rec = {"step": int(getattr(state, "global_step", 0)), "time": time.time()}
        for k, v in dict(logs).items():
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


_REWARD_CALL_ID = 0
_WARNED_TERMINAL_DEGENERATE = False
_DEGENERATE_TERMINAL_CALLS = 0


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "").strip()


def _strip_tag(text: str, tag: str) -> str:
    return re.sub(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _strip_code_blocks(text: str) -> str:
    # Remove fenced code blocks if the prover emits them (shouldn't).
    return re.sub(r"```(?:python)?\s*\n.*?\n```", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _as_bullets(block: str) -> str:
    lines = [ln.strip() for ln in (block or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    if any(ln.startswith(("-", "*")) for ln in lines):
        return "\n".join(lines)
    return "\n".join([f"- {ln}" for ln in lines])


def _bullet_ratio(text: str) -> float:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return 0.0
    bullets = sum(1 for ln in lines if ln.startswith(("-", "*")))
    return bullets / max(1, len(lines))


def _extract_identifiers_from_code(code: str, max_n: int = 30) -> List[str]:
    import keyword

    text = code or ""
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    freq: Dict[str, int] = {}
    for t in toks:
        if keyword.iskeyword(t) or t in {"True", "False", "None"}:
            continue
        freq[t] = freq.get(t, 0) + 1
    # Prefer function/class names if present.
    for m in re.finditer(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M):
        name = m.group(2)
        freq[name] = freq.get(name, 0) + 100
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _v in ranked[:max_n]]


def _clean_md_line(line: str) -> str:
    ln = (line or "").strip()
    if not ln:
        return ""
    # Drop markdown heading markers and emphasis.
    ln = re.sub(r"^\s*#+\s*", "", ln)
    ln = re.sub(r"[*_`]+", "", ln)
    # Strip common list / numbering prefixes.
    ln = re.sub(r"^\s*(?:[-*•]|\d+\s*[\.\)]|[a-zA-Z]\s*[\.\)])\s*", "", ln)
    # Strip SG labels (SG1:, Subgoal 1:, etc).
    ln = re.sub(r"^\s*(?:SG\s*\d+|Subgoal\s*\d+)\s*:?\s*", "", ln, flags=re.IGNORECASE)
    return ln.strip()


def _extract_block_by_heading(text: str, heading_pat: str) -> str:
    """
    Extracts a section that starts at a heading-like line matching `heading_pat`
    and ends at the next markdown heading or end-of-text.
    """
    t = (text or "").replace("\r\n", "\n")
    # Find a heading line and capture everything after it.
    m = re.search(rf"(?:^|\n)\s*(?:#+\s*)?{heading_pat}\s*(?:\n|$)", t, flags=re.IGNORECASE)
    if not m:
        return ""
    rest = t[m.end() :]
    # Stop at next markdown heading.
    m2 = re.search(r"(?:^|\n)\s*#+\s+.+(?:\n|$)", rest)
    return (rest[: m2.start()] if m2 else rest).strip()


def _parse_subgoals_gap_from_type_analysis(type_analysis: str) -> tuple[List[str], str]:
    ta = (type_analysis or "").replace("\r\n", "\n").strip()
    if not ta:
        return [], ""

    # DeepSeek often wraps headings in markdown (**...**, ### ...). Strip light markdown for parsing.
    ta_search = re.sub(r"[*_`]+", "", ta)

    # Subgoals: handle both the original numbered format and markdown variants.
    sub_block = ""
    # Original intended heading.
    m = re.search(
        r"(?:^|\n)\s*(?:#+\s*)?2\s*[\.\)]\s*Key\s+invariants\s*/\s*subgoals\s*:?\s*\n(.*?)(?=(?:\n\s*(?:#+\s*)?3\s*[\.\)]\s*Concise\s+gap\s+analysis|\Z))",
        ta_search,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        sub_block = (m.group(1) or "").strip()
    if not sub_block:
        sub_block = _extract_block_by_heading(ta_search, r"Key\s+Invariants\s*/\s*Subgoals")

    # Prefer an explicit "Subgoals:" subsection if present.
    if sub_block:
        m = re.search(r"(?:^|\n)\s*(?:\*\*)?\s*Subgoals?\s*(?:\*\*)?\s*:?\s*\n(.*)$", sub_block, flags=re.IGNORECASE | re.DOTALL)
        if m:
            sub_block = (m.group(1) or "").strip()

    subgoals: List[str] = []
    if sub_block:
        for raw_ln in sub_block.splitlines():
            ln = _clean_md_line(raw_ln)
            if not ln:
                continue
            # Heuristic: keep reasonably "actionable" lines.
            if len(ln) < 6:
                continue
            # Skip invariants-only labels.
            if re.match(r"^\s*Invariants?\s*:?\s*$", raw_ln, flags=re.IGNORECASE):
                continue
            subgoals.append(ln)

    # Gap analysis: prefer explicit heading; otherwise pick a conservative fallback.
    gap_block = ""
    m = re.search(
        r"(?:^|\n)\s*(?:#+\s*)?3\s*[\.\)]\s*Concise\s+gap\s+analysis\s*:?\s*\n(.*)$",
        ta_search,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        gap_block = (m.group(1) or "").strip()
    if not gap_block:
        gap_block = _extract_block_by_heading(ta_search, r"(?:Concise\s+)?Gap\s+Analysis")
    if not gap_block:
        # Many DeepSeek reasoner outputs end with a "No gaps..." style conclusion.
        m = re.search(r"(?:^|\n)\s*(No\s+gaps?.{0,2000})\s*$", ta_search, flags=re.IGNORECASE | re.DOTALL)
        if m:
            gap_block = (m.group(1) or "").strip()

    gap_lines = [_clean_md_line(ln) for ln in (gap_block or "").splitlines()]
    gap = "\n".join([ln for ln in gap_lines if ln]).strip()
    return subgoals, gap


def extract_itssm_targets(example: Dict[str, Any]) -> tuple[List[str], str, str]:
    """
    Return (subgoals_list, gap_text, checklist_text) for a dataset row.

    Many existing `itssm_lcb_v*.jsonl` files have empty `subgoals`/`gap_analysis` due to
    upstream formatting differences (e.g. markdown headings). This function derives
    missing targets from `type_analysis` as a backward-compatible fallback.
    """
    # Subgoals
    raw_sub = example.get("subgoals")
    sub_list: List[str] = []
    if isinstance(raw_sub, list):
        sub_list = [str(x).strip() for x in raw_sub if str(x).strip()]
    elif isinstance(raw_sub, str) and raw_sub.strip():
        sub_list = [ln.strip() for ln in raw_sub.splitlines() if ln.strip()]

    gap = str(example.get("gap_analysis") or "").strip()
    checklist = str(example.get("robustness_checklist") or "").strip()

    if not sub_list or not gap:
        parsed_sub, parsed_gap = _parse_subgoals_gap_from_type_analysis(str(example.get("type_analysis") or ""))
        if not sub_list:
            sub_list = parsed_sub
        if not gap:
            gap = parsed_gap

    # As a last resort, if checklist was accidentally merged into type_analysis, try to split it out.
    if not checklist:
        ta = str(example.get("type_analysis") or "")
        m = re.search(r"(?:^|\n)\s*Robustness\s+Checklist\s*:?\s*\n(.*)$", ta, flags=re.IGNORECASE | re.DOTALL)
        if m:
            checklist = (m.group(1) or "").strip()

    return sub_list, gap, checklist


def critique_quality_score(
    *,
    subgoals_block: str,
    gap_block: str,
    checklist_block: str,
    raw_completion: str,
    draft_code: str,
    min_chars: int = 120,
    max_chars: int = 1800,
) -> float:
    """
    Heuristic "good critique" score in [0, 1].
    Encourages structured, concise, grounded feedback rather than text similarity.
    """

    present = sum(1 for b in (subgoals_block, gap_block, checklist_block) if (b or "").strip())
    tag_score = present / 3.0

    sg_lines = [ln.strip() for ln in (subgoals_block or "").splitlines() if ln.strip()]
    structure = 0.4 * min(1.0, len(sg_lines) / 4.0) + 0.3 * _bullet_ratio(gap_block) + 0.3 * _bullet_ratio(checklist_block)

    total_text = "\n".join([_as_bullets(subgoals_block), _as_bullets(gap_block), _as_bullets(checklist_block)]).strip()
    L = len(total_text)
    if L <= 0:
        length_score = 0.0
    elif min_chars <= L <= max_chars:
        length_score = 1.0
    else:
        length_score = max(0.0, L / float(min_chars)) if L < min_chars else max(0.0, max_chars / float(L))

    text_low = total_text.lower()
    categories = [
        r"\binput\b|\boutput\b|\breturn\b|\bparameter",
        r"edge case|corner case|empty|single|duplicate|negative|zero",
        r"complexity|big[- ]?o|time|space|optimi",
        r"format|whitespace|newline|order|tie[- ]?break",
        r"overflow|int|float|precision|mod",
    ]
    coverage = sum(1 for pat in categories if re.search(pat, text_low)) / max(1, len(categories))

    ids = _extract_identifiers_from_code(draft_code, max_n=20)
    mentioned = 0
    if ids:
        for name in ids[:10]:
            if re.search(rf"\b{re.escape(name)}\b", total_text):
                mentioned += 1
    grounding = min(1.0, mentioned / 3.0)

    raw = raw_completion or ""
    bad = 0
    if "```" in raw:
        bad += 1
    if re.search(r"^\s*(def|class|import)\b", raw, re.M):
        bad += 1
    code_penalty = 1.0 / (1.0 + bad)

    score = 0.20 * tag_score + 0.25 * structure + 0.15 * length_score + 0.20 * coverage + 0.20 * grounding
    return float(max(0.0, min(1.0, score * code_penalty)))


class LLMJudge:
    """
    Frozen "LLM-as-a-judge" reward model for critique quality.

    Supports either:
    - OpenAI-compatible `/chat/completions` endpoints, or
    - Amplify `/chat` endpoints.
    This does not consume GPU0 VRAM if the judge is served elsewhere.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str = "",
        provider: str = "auto",
        timeout_s: int = 120,
        max_tokens: int = 256,
        strict: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = _resolve_api_key(str(api_key or ""))
        self.provider = str(provider or "auto").strip().lower()
        self.timeout_s = int(timeout_s)
        self.max_tokens = int(max_tokens)
        self.strict = bool(strict)
        if self.provider not in {"auto", "openai", "amplify"}:
            raise ValueError(f"Unsupported judge provider: {self.provider}")
        self.client = None
        if not self._is_amplify():
            try:
                import openai  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Missing dependency: `openai`") from exc
            self.client = openai.OpenAI(
                api_key=(self.api_key if self.api_key else "EMPTY"),
                base_url=self.base_url,
                timeout=self.timeout_s,
            )

        # Simple in-process cache to avoid repeated judge calls on identical inputs.
        self._cache: Dict[str, float] = {}

    def _is_amplify(self) -> bool:
        if self.provider == "amplify":
            return True
        if self.provider == "openai":
            return False
        return "vanderbilt.ai" in self.base_url

    @staticmethod
    def _extract_text_from_json_payload(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return ""
        data = payload.get("data")
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("textContent", "content", "message", "delta", "text"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                delta = c0.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        return content
        return ""

    @staticmethod
    def _hash(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        """
        Accept either JSON: {"score": 0.73} or a bare number.
        """
        s = (text or "").strip()
        if not s:
            return None
        # Try JSON first.
        try:
            j = json.loads(s)
            if isinstance(j, dict) and "score" in j:
                val = float(j["score"])
                if 0.0 <= val <= 1.0:
                    return val
        except Exception:
            pass
        # Fallback: first number in text.
        m = re.search(r"([01](?:\.\d+)?)", s)
        if not m:
            return None
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return val
        except Exception:
            return None
        return None

    def score(
        self,
        *,
        problem: str,
        starter_code: str,
        draft_code: str,
        subgoals: str,
        gap: str,
        checklist: str,
    ) -> float:
        rubric = (
            "You are a strict evaluator for a *prover/critic* in a code-generation pipeline.\n"
            "The critic must help a separate coder revise the draft solution.\n\n"
            "Score the critic output in [0,1] where:\n"
            "- 1.0 = highly actionable, grounded in the given draft/code, covers key pitfalls (I/O, edge cases, complexity, formatting), "
            "and is concise and well-structured (bullets).\n"
            "- 0.0 = generic, wrong, ungrounded, or contains code.\n\n"
            "Overconfidence penalty:\n"
            "- If it claims the draft is correct / has no issues / has no logical gaps, score it low (<=0.3) unless it ALSO provides "
            "concrete checks, potential counterexamples, and performance/formatting risks.\n\n"
            "Return ONLY valid JSON: {\"score\": <float between 0 and 1>}.\n"
        )

        user = (
            "### Problem\n" + (problem or "") + "\n\n"
            "### Starter Code\n```python\n" + (starter_code or "") + "\n```\n\n"
            "### Draft Code\n```python\n" + (draft_code or "") + "\n```\n\n"
            "### Critic Output\n"
            "<subgoal>\n" + (subgoals or "") + "\n</subgoal>\n"
            "<gap_analysis>\n" + (gap or "") + "\n</gap_analysis>\n"
            "<checklist>\n" + (checklist or "") + "\n</checklist>\n"
        )

        key = self._hash(user)
        if key in self._cache:
            return float(self._cache[key])

        text = ""
        if self._is_amplify():
            if not self.api_key:
                self.api_key = _resolve_api_key("")
            if not self.api_key:
                env_seen = bool(os.getenv("QUALITY_LLM_API_KEY") or os.getenv("AMPLIFY_API_KEY"))
                raise RuntimeError(
                    "Amplify judge selected but no API key provided. "
                    "Set QUALITY_LLM_API_KEY/AMPLIFY_API_KEY or pass --quality-llm-api-key. "
                    f"(env visible in this process: {env_seen})"
                )
            url = f"{self.base_url}/chat"
            body = {
                "data": {
                    "temperature": 0.0,
                    "max_tokens": int(self.max_tokens),
                    "type": "prompt",
                    "dataSources": [],
                    "messages": [
                        {"role": "system", "content": rubric},
                        {"role": "user", "content": user},
                    ],
                    "options": {
                        "ragOnly": False,
                        "skipRag": True,
                        "model": {"id": self.model_name},
                        "prompt": user,
                    },
                }
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
            }
            response = requests.post(
                url,
                headers=headers,
                json=body,
                stream=True,
                timeout=(15, max(30, self.timeout_s)),
            )
            try:
                response.raise_for_status()
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "application/json" in content_type and "event-stream" not in content_type:
                    payload = response.json()
                    if isinstance(payload, dict) and payload.get("success") is False:
                        raise RuntimeError(f"Amplify judge /chat error: {payload.get('message')}")
                    text = self._extract_text_from_json_payload(payload).strip()
                else:
                    chunks: List[str] = []
                    for raw in response.iter_lines(decode_unicode=True):
                        if raw is None:
                            continue
                        line = str(raw).strip()
                        if not line:
                            continue
                        if line.startswith("event:") or line.startswith("id:"):
                            continue
                        if line.startswith("data:"):
                            line = line[len("data:") :].strip()
                        if not line or line == "[DONE]":
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            chunks.append(line)
                            continue
                        piece = self._extract_text_from_json_payload(item)
                        if piece:
                            chunks.append(piece)
                    text = "".join(chunks).strip()
            finally:
                response.close()
        else:
            assert self.client is not None
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": rubric}, {"role": "user", "content": user}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()

        val = self._parse_score(text)
        if val is None:
            if self.strict:
                payload_preview = (text or "").strip().replace("\n", " ")[:500]
                raise RuntimeError(
                    "LLM judge returned an unparsable score payload. "
                    "Expected JSON like {\"score\": 0.73}. "
                    f"Payload preview: {payload_preview}"
                )
            val = 0.0
        self._cache[key] = float(val)
        return float(val)


def build_prover_stage2_prompt(problem: str, code: str) -> str:
    return f"""
You are DeepSeek Prover. Do NOT write any code.
Given the problem description and the candidate implementation, identify key subgoals, a concise gap analysis,
and a robustness checklist for a coder to fix the program.

Problem Description:
{problem}

Candidate Implementation (Python):
```python
{code}
```

Return ONLY these tags:
<subgoal>one subgoal per line</subgoal>
<gap_analysis>bullet list</gap_analysis>
<checklist>bullet list</checklist>
""".strip()


def build_coder_revision_prompt(function_signature: str, content: str, initial_code: str, type_analysis: str, starter_code: str) -> str:
    header = (
        "You are revising a Python solution to satisfy a set of logically synthesized contracts (implicit types). "
        "Return ONLY the corrected code in a single Python code block — no explanations.\n\n"
    )
    prompt = header
    prompt += f"### Original Problem\nFunction Signature: {function_signature}\n{content}\n\n"
    if starter_code:
        prompt += f"### Starter Code\n```python\n{starter_code}\n```\n\n"
    prompt += f"### Your Initial Code\n```python\n{initial_code}\n```\n\n"
    prompt += "### Synthesized Contracts and Subgoals\n" + type_analysis + "\n\n"
    prompt += (
        "### Task\nRevise the code so that it satisfies ALL contracts and subgoals. "
        "If the current code already satisfies them, return it unchanged. "
        "You may introduce small helpers if necessary, but keep the solution minimal and clear.\n\n"
    )
    prompt += "### Answer: (one Python code block, no extra text)\n"
    return prompt


def _clean_generated_code(code: str) -> str:
    code = (code or "").strip()
    if "```python" in code:
        i = code.find("```python") + len("```python")
        while i < len(code) and code[i] in " \t\r\n":
            i += 1
        j = code.find("```", i)
        code = code[i:j].strip() if j != -1 else code[i:].strip()
    elif "```" in code:
        lines = code.split("\n")
        buff: List[str] = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
            elif in_block:
                buff.append(line)
        if buff:
            code = "\n".join(buff).strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:]).strip()
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1]).strip()
    return code


def _strip_main_block(code: str) -> str:
    """
    Normalize `if __name__ == "__main__": ...` for LiveCodeBench-style execution.

    Important: Do NOT delete the guarded body (many CP solutions only call `solve()` there).
    Instead, lift the guarded body to top-level (same idea as LCB's `clean_if_name`).
    """
    src = (code or "").replace("\r\n", "\n")
    if "__main__" not in src:
        return src

    try:
        import ast

        tree = ast.parse(src)
        if not getattr(tree, "body", None):
            return src
        last = tree.body[-1]
        if isinstance(last, ast.If):
            try:
                cond = ast.unparse(last.test).strip()
            except Exception:
                cond = ""
            if cond in {"__name__ == '__main__'", '__name__ == "__main__"'}:
                tree.body = list(tree.body[:-1]) + list(last.body)
                return ast.unparse(tree) + "\n"
    except Exception:
        pass

    # Fallback: text-based lift for the common pattern (only handles 4-space indentation).
    m = re.search(
        r"(?:^|\n)if __name__\s*==\s*['\"]__main__['\"]\s*:\s*\n((?:[ \t]{4}.*(?:\n|$))+)",
        src,
        flags=re.MULTILINE,
    )
    if not m:
        return src
    body = m.group(1)
    # De-indent one level.
    lifted = re.sub(r"(?m)^[ \t]{4}", "", body).rstrip() + "\n"
    # Remove the whole guarded block and append lifted body at end.
    src2 = re.sub(
        r"(?:^|\n)if __name__\s*==\s*['\"]__main__['\"]\s*:\s*\n(?:[ \t]{4}.*(?:\n|$))+",
        "\n",
        src,
        flags=re.MULTILINE,
    ).rstrip()
    return (src2 + "\n\n" + lifted).strip() + "\n"


class QwenCoderClient:
    """OpenAI-compatible client for local vLLM Qwen endpoint (used only inside the reward)."""

    def __init__(self, *, model_name: str, base_url: str, api_key: str = "EMPTY"):
        try:
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: `openai`") from exc

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: Optional[int] = None,
    ) -> str:
        # vLLM enforces strict context limits. When the prompt is long, the requested `max_tokens`
        # can exceed (context_limit - input_tokens) and trigger a 400. Recover by retrying with a
        # reduced `max_tokens` derived from the server error message (mirrors ITSSM overflow retry).
        cur_max = int(max_tokens)
        last_exc: Optional[BaseException] = None
        for attempt in range(1, 4):
            try:
                extra_body = {"top_k": top_k, "repetition_penalty": repetition_penalty}
                if seed is not None:
                    extra_body["seed"] = int(seed)
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=cur_max,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if "maximum context length" not in msg or "too large" not in msg:
                    raise

                # Example vLLM/OpenAI-style error:
                # "... max_tokens ... is too large: 8096. This model's maximum context length is 15000 tokens
                # and your request has 7623 input tokens (8096 > 15000 - 7623)."
                m_ctx = re.search(r"maximum context length is\s*(\d+)\s*tokens", msg)
                m_inp = re.search(r"request has\s*(\d+)\s*input tokens", msg)
                if not (m_ctx and m_inp):
                    raise
                ctx = int(m_ctx.group(1))
                inp = int(m_inp.group(1))
                # Keep a small headroom to account for any chat/message framing tokens.
                margin = 256
                allowed = ctx - inp - margin
                if allowed <= 0:
                    raise RuntimeError(
                        f"Coder prompt is too long for the model context window: "
                        f"context={ctx} input={inp} margin={margin}. "
                        "Shorten the prompt (problem/draft/analysis) or use a longer-context coder."
                    ) from exc
                new_max = min(cur_max - 1, allowed)
                if new_max >= cur_max:
                    raise
                if _rank0():
                    print(f"[warn] coder context overflow at max_tokens={cur_max}; retry with {new_max}")
                cur_max = int(new_max)
                continue
        assert last_exc is not None
        raise last_exc


def _expand_jsonl_inputs(path_or_glob: Optional[str]) -> List[Path]:
    spec = (path_or_glob or "").strip()
    if not spec:
        auto = sorted(Path(".").glob("itssm_lcb_v*.jsonl"))
        if not auto:
            raise FileNotFoundError("No itssm_lcb_v*.jsonl found in CWD; pass --tasks-jsonl explicitly.")
        return [p.resolve() for p in auto]

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: List[Path] = []
    has_explicit_repeats = False
    for part_raw in parts:
        part = part_raw
        repeat = 1
        if "@" in part_raw:
            base, suffix = part_raw.rsplit("@", 1)
            if base.strip() and suffix.strip().isdigit():
                repeat = int(suffix.strip())
                part = base.strip()
                if repeat <= 0:
                    raise ValueError(f"Repeat count must be positive in: {part_raw}")
                if repeat != 1:
                    has_explicit_repeats = True

        expanded: List[Path] = []
        p = Path(part)
        if p.exists() and p.is_dir():
            expanded = [c.resolve() for c in sorted(p.glob("itssm_lcb_v*.jsonl"))]
        elif any(ch in part for ch in ["*", "?", "["]):
            expanded = [Path(m).resolve() for m in sorted(glob.glob(part)) if Path(m).is_file()]
        elif p.exists() and p.is_file():
            expanded = [p.resolve()]
        else:
            raise FileNotFoundError(f"JSONL input not found: {part_raw}")

        for _ in range(repeat):
            out.extend(expanded)

    if not out:
        raise FileNotFoundError("No JSONL inputs remain.")

    # De-dup preserve order (unless the user requested explicit repeats for weighting).
    if not has_explicit_repeats:
        seen: set[str] = set()
        uniq: List[Path] = []
        for p in out:
            k = str(p)
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        return uniq
    return out


def load_itssm_jsonl(paths: Sequence[Path], limit: Optional[int]) -> List[Dict[str, Any]]:
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


class _ConsoleTrainLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        try:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        except Exception:
            rank = 0
        if rank != 0:
            return control
        step = getattr(state, "global_step", None)
        loss = logs.get("loss", logs.get("train_loss", None))
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
        if pieces:
            print("[train] " + " ".join(pieces))
        return control


class _PeriodicCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        *,
        output_dir: Path,
        processing_class: Any,
        base_model: str,
        peft_mode: str,
        save_every_steps: int = 0,
        keep_last: int = 3,
        save_best: bool = False,
        best_metric: str = "reward",
        greater_is_better: bool = True,
        best_min_steps: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.processing_class = processing_class
        self.base_model = str(base_model or "")
        self.peft_mode = str(peft_mode or "")
        self.save_every_steps = max(0, int(save_every_steps))
        self.keep_last = max(1, int(keep_last))
        self.save_best = bool(save_best)
        self.best_metric = str(best_metric or "reward")
        self.greater_is_better = bool(greater_is_better)
        self.best_min_steps = max(0, int(best_min_steps))

        self._trainer = None
        self._model = None
        self._saved_periodic_dirs: List[Path] = []
        self._saved_periodic_steps: set[int] = set()
        self._best_value: Optional[float] = None
        self._best_step: Optional[int] = None

    def bind_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def _maybe_save_periodic(self, step: int) -> None:
        if step <= 0:
            return
        if self.save_every_steps <= 0:
            return
        if step in self._saved_periodic_steps:
            return
        if (step % self.save_every_steps) != 0:
            return
        self._saved_periodic_steps.add(step)
        self._save_checkpoint(step=step, tag=f"checkpoint-step-{step}", periodic=True)

    def _save_checkpoint(self, *, step: int, tag: str, metric_value: Optional[float] = None, periodic: bool = False) -> None:
        if not _rank0():
            return
        model = self._model
        if model is None and self._trainer is not None:
            model = getattr(self._trainer, "model", None)
        if model is None:
            return

        ckpt_dir = self.output_dir / tag
        if ckpt_dir.exists():
            # `checkpoint-best` is overwritten; step checkpoints are unique.
            if tag == "checkpoint-best":
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(ckpt_dir))
        try:
            if self.processing_class is not None:
                self.processing_class.save_pretrained(str(ckpt_dir))
        except Exception:
            pass

        meta: Dict[str, Any] = {
            "base_model": self.base_model,
            "peft": self.peft_mode,
            "script": Path(__file__).name,
            "checkpoint_step": int(step),
        }
        if metric_value is not None:
            meta["best_metric"] = self.best_metric
            meta["best_metric_value"] = float(metric_value)
        (ckpt_dir / "base_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[save] Saved checkpoint: {ckpt_dir}")

        if periodic:
            self._saved_periodic_dirs.append(ckpt_dir)
            while len(self._saved_periodic_dirs) > self.keep_last:
                old = self._saved_periodic_dirs.pop(0)
                if old != ckpt_dir and old.exists():
                    shutil.rmtree(old, ignore_errors=True)
                    print(f"[save] Pruned old checkpoint: {old}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        if not _rank0():
            return control
        if self._model is None:
            self._model = kwargs.get("model", None)
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            return control

        if self.save_best:
            if step < self.best_min_steps:
                return control
            raw_metric = logs.get(self.best_metric, None)
            if raw_metric is None:
                return control
            try:
                metric_value = float(raw_metric)
            except Exception:
                return control
            is_better = False
            if self._best_value is None:
                is_better = True
            elif self.greater_is_better:
                is_better = metric_value > self._best_value
            else:
                is_better = metric_value < self._best_value
            if is_better:
                self._best_value = float(metric_value)
                self._best_step = int(step)
                self._save_checkpoint(step=step, tag="checkpoint-best", metric_value=metric_value, periodic=False)
                print(
                    f"[save] Updated best checkpoint on metric `{self.best_metric}`: "
                    f"step={self._best_step} value={self._best_value:.6f}"
                )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not _rank0():
            return control
        if self._model is None:
            self._model = kwargs.get("model", None)
        step = int(getattr(state, "global_step", 0) or 0)
        self._maybe_save_periodic(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not _rank0():
            return control
        try:
            summary = {
                "best_metric": self.best_metric,
                "greater_is_better": self.greater_is_better,
                "best_value": self._best_value,
                "best_step": self._best_step,
                "saved_periodic_steps": sorted(int(x) for x in self._saved_periodic_steps),
            }
            fp = self.output_dir / "checkpoint_summary.json"
            fp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception:
            pass
        return control


def _is_peft_adapter_dir(path: Path) -> bool:
    return (
        path.exists()
        and (
            (path / "adapter_config.json").exists()
            or (path / "adapter_model.safetensors").exists()
            or (path / "adapter_model.bin").exists()
        )
    )


def _load_base_model_hint(adapter_dir: Path) -> Optional[str]:
    hint = adapter_dir / "base_model.json"
    if not hint.exists():
        return None
    try:
        j = json.loads(hint.read_text(encoding="utf-8"))
        base = str(j.get("base_model") or "").strip()
        return base or None
    except Exception:
        return None


def main() -> None:
    class _ArgParser(argparse.ArgumentParser):
        # Allow `@argsfile` with normal shell-style splitting per line.
        def convert_arg_line_to_args(self, arg_line: str):
            for arg in shlex.split(arg_line, comments=True, posix=True):
                arg = (arg or "").strip()
                if arg:
                    yield arg

    ap = _ArgParser(
        description="GRPO RL for DeepSeek Prover as ITSSM critic (process-aligned).",
        fromfile_prefix_chars="@",
    )
    ap.add_argument("--model-name-or-path", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    ap.add_argument(
        "--base-model-name-or-path",
        type=str,
        default=None,
        help="If --model-name-or-path is a PEFT adapter dir, load this base model (else auto-read base_model.json).",
    )
    ap.add_argument("--output-dir", type=str, default="rl_deepseek_prover_out_critic")
    ap.add_argument("--tasks-jsonl", type=str, default=None, help="Path/glob/dir/comma-list (defaults to itssm_lcb_v*.jsonl in CWD)")
    ap.add_argument("--limit", type=int, default=None)

    # Reward models
    ap.add_argument("--subgoal-reward", choices=["embedding", "rm"], default="embedding")
    ap.add_argument("--subgoal-reward-model", type=str, default="intfloat/e5-small-v2")
    ap.add_argument("--subgoal-reward-device", type=str, default="cpu")
    ap.add_argument("--subgoal-reward-max-length", type=int, default=256)

    # Reward weights
    ap.add_argument("--dense-weight", type=float, default=0.3)
    ap.add_argument("--terminal-weight", type=float, default=0.7)
    ap.add_argument("--dense-subgoal-weight", type=float, default=1.0)
    ap.add_argument("--dense-gap-weight", type=float, default=1.0)
    ap.add_argument("--dense-checklist-weight", type=float, default=0.5)
    ap.add_argument("--dense-embed-weight", type=float, default=0.2, help="Weight for embedding similarity inside dense reward")
    ap.add_argument("--dense-quality-weight", type=float, default=0.8, help="Weight for heuristic critique quality inside dense reward")
    ap.add_argument(
        "--quality-reward",
        choices=["heuristic", "llm"],
        default="llm",
        help="Critique-quality reward backend. `llm` uses a strong frozen judge (recommended if available).",
    )
    ap.add_argument(
        "--quality-llm-base-url",
        type=str,
        default=os.getenv("QUALITY_LLM_BASE_URL", "https://prod-api.vanderbilt.ai"),
        help="(quality-reward=llm) Judge base URL (defaults to Amplify endpoint).",
    )
    ap.add_argument(
        "--quality-llm-provider",
        choices=["auto", "openai", "amplify"],
        default=os.getenv("QUALITY_LLM_PROVIDER", "amplify"),
        help="Judge backend: OpenAI-compatible or Amplify /chat.",
    )
    ap.add_argument(
        "--quality-llm-model",
        type=str,
        default=os.getenv("QUALITY_LLM_MODEL", "gpt-5.2"),
        help="(quality-reward=llm) Judge model name.",
    )
    ap.add_argument(
        "--quality-llm-api-key",
        type=str,
        default=_resolve_api_key(""),
        help=(
            "(quality-reward=llm) Optional API key for judge endpoint "
            "(required for Amplify). Falls back to QUALITY_LLM_API_KEY/AMPLIFY_API_KEY env vars."
        ),
    )
    ap.add_argument(
        "--allow-local-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow using the same local endpoint/model for coder and judge. "
            "Disabled by default to avoid reward leakage from self-judging."
        ),
    )
    ap.add_argument(
        "--quality-llm-max-tokens",
        type=int,
        default=256,
        help="(quality-reward=llm) Max tokens for judge response (JSON only).",
    )
    ap.add_argument(
        "--quality-llm-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "(quality-reward=llm) If true, abort training when judge output is invalid/unparsable "
            "instead of silently assigning zero."
        ),
    )

    # GRPO knobs
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.05, help="KL strength")
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--min-present-target-tags",
        type=int,
        default=2,
        help="Minimum number of non-empty targets among {subgoals,gap_analysis,checklist} required to keep a row.",
    )
    ap.add_argument("--num-generations", type=int, default=2)
    ap.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help="Lower this to reduce peak VRAM during sampling (often the #1 knob for cublas/OOM).",
    )
    ap.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=0,
        help="If >0, call torch.cuda.empty_cache() every N steps (slower, but can reduce fragmentation).",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)

    # Coder inside reward
    ap.add_argument(
        "--terminal-mode",
        choices=["coder", "draft", "none"],
        default="coder",
        help="How to compute terminal reward: via coder revision + tests (coder), tests on draft_code (draft), or disable (none).",
    )
    ap.add_argument(
        "--terminal-relative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use relative terminal reward: score(with_critique) - score(baseline_no_critique). Recommended for generalization.",
    )
    ap.add_argument(
        "--terminal-baseline",
        choices=["draft", "empty", "generic"],
        default="draft",
        help="Baseline for relative terminal reward: `draft` skips the coder and tests the original draft; `empty`/`generic` run coder with an empty/generic critique.",
    )
    ap.add_argument(
        "--terminal-abs-weight",
        type=float,
        default=0.0,
        help="Optional absolute terminal shaping: terminal += terminal_abs_weight * score_with. Use small values (e.g. 0.05).",
    )
    ap.add_argument(
        "--terminal-runtime-penalty",
        type=float,
        default=0.02,
        help="Extra penalty subtracted when revised code crashes with RE (error_code=-4). Helps avoid all-zero RE batches.",
    )
    ap.add_argument(
        "--dense-no-terminal-scale",
        type=float,
        default=0.1,
        help=(
            "Scale dense reward when relative terminal signal is absent "
            "(score_with≈score_base≈0). Use 0.0 to disable dense reward on no-signal rows."
        ),
    )
    ap.add_argument(
        "--no-terminal-signal-eps",
        type=float,
        default=1e-9,
        help="Numerical epsilon for detecting no terminal signal in relative mode.",
    )
    ap.add_argument(
        "--skip-baseline-perfect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, set reward=0 for rows where relative baseline score is already perfect (>= --baseline-perfect-threshold).",
    )
    ap.add_argument(
        "--baseline-perfect-threshold",
        type=float,
        default=0.999999,
        help="Threshold used by --skip-baseline-perfect.",
    )
    ap.add_argument(
        "--abort-on-degenerate-terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort training if terminal rewards stay identically zero for too many reward calls.",
    )
    ap.add_argument(
        "--degenerate-terminal-max-calls",
        type=int,
        default=10,
        help="If --abort-on-degenerate-terminal, stop after this many fully-degenerate reward calls.",
    )
    ap.add_argument("--qwen-base-url", type=str, default=os.getenv("QWEN_VLLM_BASE_URL", "http://localhost:1234/v1"))
    ap.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    ap.add_argument("--coder-max-tokens", type=int, default=2048)
    # Match evaluation defaults (ITSSM script uses ~0.6).
    ap.add_argument("--coder-temperature", type=float, default=0.6)
    ap.add_argument("--coder-top-p", type=float, default=0.95)
    ap.add_argument("--coder-top-k", type=int, default=20)
    ap.add_argument("--coder-repetition-penalty", type=float, default=1.05)
    ap.add_argument(
        "--coder-seed",
        type=int,
        default=None,
        help="Optional fixed seed for coder sampling inside terminal reward (reduces reward variance).",
    )
    ap.add_argument("--coder-randomize", action="store_true", help="Randomize coder decoding per episode (reduces overfitting).")
    ap.add_argument("--coder-temperature-min", type=float, default=0.2)
    ap.add_argument("--coder-temperature-max", type=float, default=0.8)
    ap.add_argument("--coder-top-p-min", type=float, default=0.85)
    ap.add_argument("--coder-top-p-max", type=float, default=0.98)

    # Tests
    ap.add_argument("--test-timeout-s", type=int, default=6)
    ap.add_argument(
        "--terminal-score",
        choices=["binary", "fraction", "shaped"],
        default="shaped",
        help="How to score tests. `shaped` uses LCB metadata (WA/RE/TLE) + output similarity to avoid all-zero rewards.",
    )

    # Memory adaptation
    ap.add_argument("--peft", choices=["none", "lora", "qlora"], default="qlora")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-bias", type=str, default="none", choices=["none", "all", "lora_only"])
    ap.add_argument("--lora-target-modules", type=str, default="all-linear")
    ap.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true")

    # Saving
    ap.add_argument("--no-save-checkpoints", action="store_true", default=True, help="Always disable intermediate checkpoints (default: true)")
    ap.add_argument(
        "--enable-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic checkpoint hooks (independent of --no-save-checkpoints).",
    )
    ap.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=20,
        help="When checkpoint hooks are enabled, save a checkpoint every N training steps.",
    )
    ap.add_argument(
        "--checkpoint-keep-last",
        type=int,
        default=3,
        help="When checkpoint hooks are enabled, keep only the latest K periodic checkpoints.",
    )
    ap.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When checkpoint hooks are enabled, maintain `checkpoint-best` using --best-checkpoint-metric.",
    )
    ap.add_argument(
        "--best-checkpoint-metric",
        type=str,
        default="reward",
        help="Metric key from trainer logs to track best checkpoint (e.g., reward, rewards/reward_func/mean).",
    )
    ap.add_argument(
        "--best-checkpoint-greater-is-better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Direction for --best-checkpoint-metric optimization.",
    )
    ap.add_argument(
        "--best-checkpoint-min-steps",
        type=int,
        default=20,
        help="Do not update `checkpoint-best` before this global step.",
    )
    ap.add_argument("--no-save-final", action="store_true", help="Do not save final adapter/model")

    ap.add_argument("--log-dir", type=str, default="logs")
    ap.add_argument("--log-file", type=str, default="", help="Optional explicit log file path (default: auto under --log-dir)")
    ap.add_argument("--no-log-file", action="store_true", help="Disable saving logs to a file")
    ap.add_argument("--metrics-jsonl", type=str, default="", help="Optional JSONL metrics file path (default: auto under --log-dir)")
    ap.add_argument("--no-metrics-jsonl", action="store_true", help="Disable saving structured metrics JSONL")
    ap.add_argument(
        "--reward-metrics-jsonl",
        type=str,
        default="",
        help="Optional JSONL file for per-reward-function summaries (default: auto under --log-dir).",
    )
    ap.add_argument("--no-reward-metrics-jsonl", action="store_true", help="Disable saving reward summary JSONL")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log_fh = None
    if not args.no_log_file:
        if args.log_file.strip():
            log_path = Path(args.log_file).expanduser().resolve()
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = Path(args.log_dir) / f"grpo_prover_{ts}.log"
        log_fh = _setup_logging(log_path)
    else:
        _setup_logging(None)

    metrics_fh = None
    metrics_cb = None
    if not args.no_metrics_jsonl and _rank0():
        if args.metrics_jsonl.strip():
            mp = Path(args.metrics_jsonl).expanduser().resolve()
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            mp = Path(args.log_dir) / f"grpo_metrics_{ts}.jsonl"
        mp.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = mp.open("w", encoding="utf-8", buffering=1)
        print(f"[metrics] Writing JSONL metrics to {mp}")
        metrics_cb = _JSONLMetricsCallback(metrics_fh)

    reward_metrics_fh = None
    if not args.no_reward_metrics_jsonl and _rank0():
        if args.reward_metrics_jsonl.strip():
            rp = Path(args.reward_metrics_jsonl).expanduser().resolve()
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            rp = Path(args.log_dir) / f"reward_metrics_{ts}.jsonl"
        rp.parent.mkdir(parents=True, exist_ok=True)
        reward_metrics_fh = rp.open("w", encoding="utf-8", buffering=1)
        print(f"[metrics] Writing reward summaries to {rp}")

    # bitsandbytes 4-bit GEMMs can fail with BF16 activations on some CUDA/cuBLAS stacks.
    # For QLoRA, force FP16 to avoid `CUBLAS_STATUS_NOT_SUPPORTED` failures.
    if args.peft == "qlora" and args.bf16:
        print("[warn] --bf16 + --peft qlora is not supported on this setup; forcing fp16 for QLoRA.")
        args.bf16 = False

    if args.num_generations <= 0:
        raise SystemExit("--num-generations must be >= 1")
    if args.generation_batch_size <= 0:
        raise SystemExit("--generation-batch-size must be >= 1")
    if args.checkpoint_every_steps <= 0 and (args.enable_checkpoints or args.save_best_checkpoint):
        raise SystemExit("--checkpoint-every-steps must be >= 1 when checkpoint hooks are enabled.")
    if args.checkpoint_keep_last <= 0:
        raise SystemExit("--checkpoint-keep-last must be >= 1")
    if args.baseline_perfect_threshold < 0.0 or args.baseline_perfect_threshold > 1.0:
        raise SystemExit("--baseline-perfect-threshold must be in [0,1]")
    if args.quality_reward == "llm":
        judge_model_required = "gpt-5.2"
        judge_model = str(args.quality_llm_model or "").strip()
        if judge_model != judge_model_required:
            raise SystemExit(
                f"Only `{judge_model_required}` is allowed as RL judge model. "
                f"Got --quality-llm-model `{judge_model}`."
            )
        judge_provider = str(args.quality_llm_provider or "").strip().lower()
        if judge_provider not in {"amplify", "auto"}:
            raise SystemExit(
                "RL judge provider must be `amplify` (or `auto` resolving to Amplify) "
                f"for GPT-5.2. Got --quality-llm-provider `{judge_provider}`."
            )
        judge_base_url = str(args.quality_llm_base_url or "").strip()
        if "vanderbilt.ai" not in judge_base_url:
            raise SystemExit(
                "RL judge must use Amplify endpoint. "
                "Set --quality-llm-base-url to `https://prod-api.vanderbilt.ai`."
            )
        if judge_provider == "auto":
            args.quality_llm_provider = "amplify"
        if _rank0():
            print(
                "[judge] Enforced judge config: provider=amplify model=gpt-5.2 "
                f"base_url={judge_base_url} strict={bool(args.quality_llm_strict)}"
            )
    if args.skip_baseline_perfect and _rank0():
        print(
            "[reward] Baseline-perfect gating enabled: "
            f"score_base >= {float(args.baseline_perfect_threshold):.6f} => reward=0 for that row."
        )
    # TRL constraint: generation_batch_size must be divisible by num_generations.
    # For peak VRAM, the smallest valid value is `num_generations`.
    if args.generation_batch_size % args.num_generations != 0:
        fixed = int(args.num_generations)
        print(
            f"[warn] generation_batch_size ({args.generation_batch_size}) must be divisible by num_generations ({args.num_generations}); "
            f"using generation_batch_size={fixed}."
        )
        args.generation_batch_size = fixed

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    jsonl_paths = _expand_jsonl_inputs(args.tasks_jsonl)
    print(f"[data] Loading {len(jsonl_paths)} JSONL(s): " + ", ".join([p.name for p in jsonl_paths]))
    raw = load_itssm_jsonl(jsonl_paths, limit=args.limit)
    if not raw:
        raise SystemExit("No training rows loaded.")
    print(f"[data] Loaded {len(raw)} rows")

    # Build HF dataset
    from datasets import Dataset  # type: ignore

    rows: List[Dict[str, Any]] = []
    skipped_empty_targets = 0
    skipped_low_quality_targets = 0
    for ex in raw:
        problem = (ex.get("question_content") or ex.get("problem") or "").strip()
        starter = (ex.get("starter_code") or "").strip()
        draft = (ex.get("draft_code") or ex.get("initial_proof_term") or "").strip()
        prompt = build_prover_stage2_prompt(problem, draft)

        row: Dict[str, Any] = dict(ex)
        # ensure flat fields
        if isinstance(row.get("metadata"), dict):
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
        row["prompt"] = prompt
        row["question_content"] = problem
        row["starter_code"] = starter
        row["draft_code"] = draft
        # Backfill missing training targets from `type_analysis` if needed.
        try:
            sub_list, gap_text, checklist_text = extract_itssm_targets(row)
            if not (isinstance(row.get("subgoals"), list) and row.get("subgoals")):
                row["subgoals"] = sub_list
            if not str(row.get("gap_analysis") or "").strip():
                row["gap_analysis"] = gap_text
            if not str(row.get("robustness_checklist") or "").strip():
                row["robustness_checklist"] = checklist_text
        except Exception:
            # Keep original fields if parsing fails.
            pass
        present_tags = (
            int(isinstance(row.get("subgoals"), list) and len(row.get("subgoals") or []) > 0)
            + int(bool(str(row.get("gap_analysis") or "").strip()))
            + int(bool(str(row.get("robustness_checklist") or "").strip()))
        )
        if present_tags <= 0:
            skipped_empty_targets += 1
            continue
        if present_tags < int(args.min_present_target_tags):
            skipped_low_quality_targets += 1
            continue
        rows.append(row)
    if not rows:
        raise SystemExit(
            "No training rows left after target fallback + quality filtering. "
            "Lower --min-present-target-tags or regenerate dataset targets."
        )
    train_dataset = Dataset.from_list(rows)
    try:
        n = len(rows)
        n_sub = sum(1 for r in rows if isinstance(r.get("subgoals"), list) and len(r.get("subgoals") or []) > 0)
        n_gap = sum(1 for r in rows if str(r.get("gap_analysis") or "").strip())
        n_chk = sum(1 for r in rows if str(r.get("robustness_checklist") or "").strip())
        print(f"[data] Targets after fallback: subgoals_nonempty={n_sub}/{n} gap_nonempty={n_gap}/{n} checklist_nonempty={n_chk}/{n}")
        if skipped_empty_targets:
            print(f"[data] Dropped {skipped_empty_targets} rows with empty targets after fallback")
        if skipped_low_quality_targets:
            print(
                f"[data] Dropped {skipped_low_quality_targets} rows with < {int(args.min_present_target_tags)} "
                "non-empty target tags (quality filter)"
            )
    except Exception:
        pass

    # Scorer
    reward_device = torch.device(args.subgoal_reward_device)
    if args.subgoal_reward == "embedding":
        scorer: SubgoalAlignmentScorer = EmbeddingCosineScorer(
            args.subgoal_reward_model, device=reward_device, max_length=args.subgoal_reward_max_length
        )
    else:
        scorer = FrozenRewardModelScorer(
            args.subgoal_reward_model, device=reward_device, max_length=args.subgoal_reward_max_length
        )

    # Coder client (used in reward)
    coder = None
    if args.terminal_mode == "coder":
        coder = QwenCoderClient(model_name=args.qwen_model, base_url=args.qwen_base_url, api_key="EMPTY")
        print(f"[remote] coder provider=openai base_url={args.qwen_base_url} model={args.qwen_model}")

    # Test executor (public tests by default in input_output)
    executor = LiveCodeBenchExecutor(timeout_s=int(args.test_timeout_s))
    baseline_cache: Dict[str, Dict[str, Any]] = {}

    judge: Optional[LLMJudge] = None
    if args.quality_reward == "llm":
        judge_base_url = str(args.quality_llm_base_url or args.qwen_base_url).strip()
        judge_model_name = str(args.quality_llm_model or args.qwen_model).strip()
        judge_api_key = _resolve_api_key(str(args.quality_llm_api_key or ""))
        judge = LLMJudge(
            base_url=judge_base_url,
            model_name=judge_model_name,
            api_key=judge_api_key,
            provider=str(args.quality_llm_provider),
            max_tokens=int(args.quality_llm_max_tokens),
            strict=bool(args.quality_llm_strict),
        )
        try:
            judge_kind = "amplify" if judge._is_amplify() else "openai"
        except Exception:
            judge_kind = str(args.quality_llm_provider)
        print(
            f"[remote] judge provider={judge_kind} base_url={judge_base_url} model={judge_model_name} "
            f"api_key_present={bool(judge_api_key)} strict={bool(args.quality_llm_strict)}"
        )
        if judge_kind == "amplify" and not judge_api_key:
            env_seen = bool(_resolve_api_key(""))
            raise SystemExit(
                "Amplify judge selected but no API key provided. "
                "Set QUALITY_LLM_API_KEY/AMPLIFY_API_KEY, place key in .amplify_api_key, "
                "or pass --quality-llm-api-key. "
                f"(key visible in this process: {env_seen})"
            )
        try:
            same_local_endpoint = (
                judge_kind == "openai"
                and str(judge_base_url).rstrip("/") == str(args.qwen_base_url).rstrip("/")
                and str(judge_model_name).strip() == str(args.qwen_model).strip()
            )
            if same_local_endpoint:
                msg = (
                    "Judge is configured to the same local Qwen endpoint/model as coder. "
                    "Use Amplify GPT-5.2 via "
                    "`--quality-llm-provider amplify --quality-llm-base-url https://prod-api.vanderbilt.ai "
                    "--quality-llm-model gpt-5.2`."
                )
                if bool(args.allow_local_judge):
                    print(f"[warn] {msg} Continuing because --allow-local-judge is set.")
                else:
                    raise SystemExit(msg + " Refusing to continue without --allow-local-judge.")
        except Exception:
            pass

    # Policy model + PEFT
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    peft_config = None

    model_path = Path(args.model_name_or_path)
    is_adapter = _is_peft_adapter_dir(model_path)
    base_model_name = args.base_model_name_or_path or (_load_base_model_hint(model_path) if is_adapter else None)
    if is_adapter and not base_model_name:
        raise SystemExit(
            "Detected a PEFT adapter directory for --model-name-or-path but base model is unknown. "
            "Pass --base-model-name-or-path or ensure base_model.json exists in the adapter dir."
        )

    device_map = None
    if args.peft == "qlora":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            device_map = {"": local_rank}

        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        base_for_load = base_model_name or args.model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            base_for_load,
            trust_remote_code=True,
            quantization_config=qconf,
            torch_dtype=dtype,
            device_map=device_map,
        )
        try:
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        except Exception:
            pass
    else:
        base_for_load = base_model_name or args.model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(base_for_load, trust_remote_code=True, torch_dtype=dtype)

    # If we are warm-starting from an adapter directory, load it and continue training it.
    if is_adapter:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, args.model_name_or_path, is_trainable=True)
    else:
        # Otherwise create a fresh adapter if requested.
        if args.peft in ("lora", "qlora"):
            from peft import LoraConfig, TaskType  # type: ignore

            target_modules: Any
            if args.lora_target_modules.strip().lower() == "all-linear":
                target_modules = "all-linear"
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

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    else:
        # Even without checkpointing, disabling KV-cache can significantly reduce peak VRAM during long generation.
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    tok_source = args.model_name_or_path
    if is_adapter and not (model_path / "tokenizer_config.json").exists():
        tok_source = base_model_name or args.model_name_or_path
    processing_class = AutoTokenizer.from_pretrained(tok_source, use_fast=True, trust_remote_code=True)
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    # Reward function: completion -> build critic text -> coder revision -> tests
    def reward_func(prompts: List[str], completions: List[str], completion_ids=None, **kwargs) -> List[float]:
        problems = kwargs.get("question_content") or [""] * len(completions)
        starters = kwargs.get("starter_code") or [""] * len(completions)
        drafts = kwargs.get("draft_code") or [""] * len(completions)
        input_outputs = kwargs.get("input_output") or [None] * len(completions)
        question_ids = kwargs.get("question_id") or [""] * len(completions)

        ref_subgoals_col = kwargs.get("subgoals") or None
        ref_gap_col = kwargs.get("gap_analysis") or None
        ref_check_col = kwargs.get("robustness_checklist") or None

        def _extract_error_code(tr) -> Optional[int]:
            try:
                meta = (tr.details or {}).get("meta", None)
                if isinstance(meta, dict) and "error_code" in meta:
                    return int(meta["error_code"])
            except Exception:
                return None
            return None

        def _extract_error_message(tr) -> str:
            try:
                meta = (tr.details or {}).get("meta", None)
                if isinstance(meta, dict):
                    msg = meta.get("error") or meta.get("error_message")
                    if msg:
                        s = str(msg).replace("\n", " ").strip()
                        return s[:240]
            except Exception:
                return ""
            return ""

        def _output_similarity(tr) -> float:
            try:
                meta = (tr.details or {}).get("meta", None)
                if not isinstance(meta, dict):
                    return 0.0
                out = str(meta.get("output", "") or "")
                exp = str(meta.get("expected", "") or "")
                if not out or not exp:
                    return 0.0
                out = out.strip()[:400]
                exp = exp.strip()[:400]
                if not out or not exp:
                    return 0.0
                return float(difflib.SequenceMatcher(None, out, exp).ratio())
            except Exception:
                return 0.0

        def test_score(tr) -> float:
            try:
                res = (tr.details or {}).get("results", None)
                if isinstance(res, list) and res:
                    return float(sum(1 for x in res if x is True) / len(res))
            except Exception:
                pass
            return 1.0 if getattr(tr, "passed", False) else 0.0

        def terminal_score(tr) -> float:
            """
            More informative scoring than pure pass/fail:
            - uses fraction of tests passed when available
            - uses WA/RE/TLE metadata to shape a small non-zero signal
            - for WA, adds a tiny output-similarity term
            """
            mode = str(args.terminal_score)
            if mode == "binary":
                return 1.0 if getattr(tr, "passed", False) else 0.0
            frac = test_score(tr)
            if mode == "fraction":
                return float(frac)

            # shaped
            if getattr(tr, "passed", False):
                return 1.0
            err = _extract_error_code(tr)
            # Base small signal by failure type (LCB uses: WA=-2, TLE=-3, RE=-4).
            base = 0.0
            if err == -2:
                base = 0.05
            elif err == -3:
                base = 0.02
            elif err == -4:
                base = 0.0
            sim = _output_similarity(tr) if err == -2 else 0.0
            # Weighting: keep shaped score in ~[0, 1].
            return float(max(0.0, min(1.0, 0.85 * frac + base + 0.1 * sim)))

        rewards: List[float] = []
        dense_terms: List[float] = []
        dense_raw_terms: List[float] = []
        terminal_terms: List[float] = []
        embed_terms: List[float] = []
        quality_terms: List[float] = []
        score_with_terms: List[float] = []
        score_base_terms: List[Optional[float]] = []
        with_err_counts: Dict[str, int] = {}
        base_err_counts: Dict[str, int] = {}
        with_err_msgs: Dict[str, int] = {}
        base_err_msgs: Dict[str, int] = {}
        no_terminal_signal_rows = 0
        skipped_baseline_perfect_rows = 0
        batch_qids: List[str] = []
        for i, text in enumerate(completions):
            problem = str(problems[i] or "").strip()
            starter = str(starters[i] or "").strip()
            draft = str(drafts[i] or "").strip()
            qid = str(question_ids[i] or "").strip() if i < len(question_ids) else ""
            if qid:
                batch_qids.append(qid)

            # Parse critic tags.
            cleaned = _strip_code_blocks(_strip_tag(text, "final_code"))
            gen_subgoals = _extract_tag(cleaned, "subgoal")
            gen_gap = _extract_tag(cleaned, "gap_analysis")
            gen_check = _extract_tag(cleaned, "checklist")

            # Dense reward = embedding alignment (small weight) + critique quality (main weight).
            dense_sub = 0.0
            if isinstance(ref_subgoals_col, list) and i < len(ref_subgoals_col) and isinstance(ref_subgoals_col[i], list):
                intended = [str(x) for x in ref_subgoals_col[i] if str(x).strip()]
                sgs = [ln.strip() for ln in gen_subgoals.splitlines() if ln.strip()]
                if sgs and intended:
                    sub_sum = 0.0
                    for j, sg in enumerate(sgs[: len(intended)]):
                        sub_sum += scorer.score(problem=problem, generated_step=sg, intended_subgoal=intended[j])
                    dense_sub = sub_sum / max(1, min(len(sgs), len(intended)))

            dense_gap = 0.0
            if isinstance(ref_gap_col, list) and i < len(ref_gap_col) and ref_gap_col[i] and gen_gap:
                dense_gap = scorer.score(problem=problem, generated_step=gen_gap, intended_subgoal=str(ref_gap_col[i]))

            dense_check = 0.0
            if isinstance(ref_check_col, list) and i < len(ref_check_col) and ref_check_col[i] and gen_check:
                dense_check = scorer.score(problem=problem, generated_step=gen_check, intended_subgoal=str(ref_check_col[i]))

            embed_dense = (
                args.dense_subgoal_weight * dense_sub
                + args.dense_gap_weight * dense_gap
                + args.dense_checklist_weight * dense_check
            )
            if args.quality_reward == "llm":
                assert judge is not None
                quality = judge.score(
                    problem=problem,
                    starter_code=starter,
                    draft_code=draft,
                    subgoals=gen_subgoals,
                    gap=gen_gap,
                    checklist=gen_check,
                )
            else:
                quality = critique_quality_score(
                    subgoals_block=gen_subgoals,
                    gap_block=gen_gap,
                    checklist_block=gen_check,
                    raw_completion=text,
                    draft_code=draft,
                )
            dense = args.dense_embed_weight * embed_dense + args.dense_quality_weight * quality

            # Sparse terminal: run tests either on coder-revised code (preferred) or on draft code.
            task_for_exec: Dict[str, Any] = {"question_content": problem, "starter_code": starter}
            if input_outputs[i] is not None:
                task_for_exec["input_output"] = input_outputs[i]

            terminal = 0.0
            score_with: Optional[float] = None
            score_base: Optional[float] = None
            err_with: Optional[int] = None
            err_base: Optional[int] = None
            if args.terminal_mode == "none" or args.terminal_weight <= 0.0:
                terminal = 0.0
            elif args.terminal_mode == "draft":
                tr = executor.run_tests(task=task_for_exec, solution_code=draft)
                score_with = terminal_score(tr)
                err_with = _extract_error_code(tr)
                code_with = str(err_with)
                with_err_counts[code_with] = with_err_counts.get(code_with, 0) + 1
                msg_with = _extract_error_message(tr)
                if msg_with:
                    with_err_msgs[msg_with] = with_err_msgs.get(msg_with, 0) + 1
                terminal = float(score_with)
            else:
                assert coder is not None
                # Optionally randomize coder decoding to reduce overfitting to a single coder mode.
                coder_temperature = float(args.coder_temperature)
                coder_top_p = float(args.coder_top_p)
                coder_seed = int(args.coder_seed) if args.coder_seed is not None else None
                if args.coder_randomize:
                    coder_temperature = random.uniform(float(args.coder_temperature_min), float(args.coder_temperature_max))
                    coder_top_p = random.uniform(float(args.coder_top_p_min), float(args.coder_top_p_max))

                # Build combined analysis in the exact structure the coder revision prompt expects.
                analysis = "1. Preconditions and postconditions\n- (omitted)\n\n2. Key invariants / subgoals\n"
                analysis += _as_bullets(gen_subgoals) or "- (none)\n"
                analysis += "\n\n3. Concise gap analysis\n"
                analysis += _as_bullets(gen_gap) or "- (none)\n"
                analysis += "\n\nRobustness Checklist:\n"
                analysis += _as_bullets(gen_check) or "- (none)\n"

                # Build function signature (same heuristic as ITSSM stage3).
                function_signature = ""
                if starter:
                    for line in starter.split("\n"):
                        t = line.strip()
                        if t.startswith("def ") or t.startswith("class "):
                            function_signature = t
                            break

                coder_prompt = build_coder_revision_prompt(
                    function_signature=function_signature,
                    content=problem,
                    initial_code=draft,
                    type_analysis=analysis,
                    starter_code=starter,
                )
                revised = coder.generate(
                    coder_prompt,
                    max_tokens=int(args.coder_max_tokens),
                    temperature=coder_temperature,
                    top_p=coder_top_p,
                    top_k=int(args.coder_top_k),
                    repetition_penalty=float(args.coder_repetition_penalty),
                    seed=coder_seed,
                )
                revised = _strip_main_block(_clean_generated_code(revised))
                tr = executor.run_tests(task=task_for_exec, solution_code=revised)
                score_with = terminal_score(tr)
                err_with = _extract_error_code(tr)
                code_with = str(err_with)
                with_err_counts[code_with] = with_err_counts.get(code_with, 0) + 1
                msg_with = _extract_error_message(tr)
                if msg_with:
                    with_err_msgs[msg_with] = with_err_msgs.get(msg_with, 0) + 1

                if args.terminal_relative:
                    if args.terminal_baseline == "draft":
                        key = json.dumps(
                            {
                                "p": problem,
                                "s": starter,
                                "d": draft,
                                "io": input_outputs[i] if input_outputs[i] is not None else "",
                                "b": "draft",
                            },
                            sort_keys=True,
                        )
                        if key in baseline_cache:
                            cached = baseline_cache[key]
                            score_base = float(cached.get("score", 0.0))
                            err_base = cached.get("error_code")
                            code_base = str(err_base)
                            base_err_counts[code_base] = base_err_counts.get(code_base, 0) + 1
                            msg_base = str(cached.get("error_message") or "").strip()
                            if msg_base:
                                base_err_msgs[msg_base] = base_err_msgs.get(msg_base, 0) + 1
                        else:
                            tr0 = executor.run_tests(task=task_for_exec, solution_code=draft)
                            score_base = terminal_score(tr0)
                            err_base = _extract_error_code(tr0)
                            msg_base = _extract_error_message(tr0)
                            baseline_cache[key] = {
                                "score": float(score_base),
                                "error_code": err_base,
                                "error_message": msg_base,
                            }
                            code_base = str(err_base)
                            base_err_counts[code_base] = base_err_counts.get(code_base, 0) + 1
                            if msg_base:
                                base_err_msgs[msg_base] = base_err_msgs.get(msg_base, 0) + 1
                    else:
                        if args.terminal_baseline == "empty":
                            baseline_analysis = ""
                        else:
                            baseline_analysis = (
                                "1. Preconditions and postconditions\n- (none)\n\n"
                                "2. Key invariants / subgoals\n- (none)\n\n"
                                "3. Concise gap analysis\n- (none)\n\n"
                                "Robustness Checklist:\n- (none)\n"
                            )

                        tr0 = None
                        key = json.dumps(
                            {
                                "p": problem,
                                "s": starter,
                                "d": draft,
                                "io": input_outputs[i] if input_outputs[i] is not None else "",
                                "b": args.terminal_baseline,
                            },
                            sort_keys=True,
                        )
                        if key in baseline_cache:
                            cached = baseline_cache[key]
                            score_base = float(cached.get("score", 0.0))
                            err_base = cached.get("error_code")
                            code_base = str(err_base)
                            base_err_counts[code_base] = base_err_counts.get(code_base, 0) + 1
                            msg_base = str(cached.get("error_message") or "").strip()
                            if msg_base:
                                base_err_msgs[msg_base] = base_err_msgs.get(msg_base, 0) + 1
                        else:
                            base_prompt = build_coder_revision_prompt(
                                function_signature=function_signature,
                                content=problem,
                                initial_code=draft,
                                type_analysis=baseline_analysis,
                                starter_code=starter,
                            )
                            base_code = coder.generate(
                                base_prompt,
                                max_tokens=int(args.coder_max_tokens),
                                temperature=coder_temperature,
                                top_p=coder_top_p,
                                top_k=int(args.coder_top_k),
                                repetition_penalty=float(args.coder_repetition_penalty),
                                seed=coder_seed,
                            )
                            base_code = _strip_main_block(_clean_generated_code(base_code))
                            tr0 = executor.run_tests(task=task_for_exec, solution_code=base_code)
                            score_base = terminal_score(tr0)
                            err_base = _extract_error_code(tr0)
                            msg_base = _extract_error_message(tr0)
                            baseline_cache[key] = {
                                "score": float(score_base),
                                "error_code": err_base,
                                "error_message": msg_base,
                            }
                            code_base = str(err_base)
                            base_err_counts[code_base] = base_err_counts.get(code_base, 0) + 1
                            if msg_base:
                                base_err_msgs[msg_base] = base_err_msgs.get(msg_base, 0) + 1

                    terminal = float(score_with - float(score_base))
                else:
                    terminal = float(score_with)

            if args.terminal_runtime_penalty and err_with == -4:
                terminal = float(terminal - float(args.terminal_runtime_penalty))

            if args.terminal_abs_weight and score_with is not None:
                terminal = float(terminal + float(args.terminal_abs_weight) * float(score_with))

            dense_raw = float(dense)
            dense_effective = dense_raw
            if (
                bool(args.terminal_relative)
                and score_with is not None
                and score_base is not None
            ):
                eps = float(args.no_terminal_signal_eps)
                delta = float(score_with) - float(score_base)
                if abs(delta) <= eps and max(abs(float(score_with)), abs(float(score_base))) <= eps:
                    dense_effective = float(dense_effective * float(args.dense_no_terminal_scale))
                    no_terminal_signal_rows += 1

            skip_baseline_perfect = (
                bool(args.skip_baseline_perfect)
                and bool(args.terminal_relative)
                and score_base is not None
                and float(score_base) >= float(args.baseline_perfect_threshold)
            )
            if skip_baseline_perfect:
                skipped_baseline_perfect_rows += 1
                tot = 0.0
            else:
                tot = args.dense_weight * dense_effective + args.terminal_weight * terminal
            rewards.append(tot)
            dense_terms.append(float(dense_effective))
            dense_raw_terms.append(float(dense_raw))
            terminal_terms.append(float(terminal))
            embed_terms.append(float(embed_dense))
            quality_terms.append(float(quality))
            score_with_terms.append(float(score_with) if score_with is not None else 0.0)
            score_base_terms.append(float(score_base) if score_base is not None else None)

        # Write a compact reward summary for debugging/plotting (rank0 only).
        global _REWARD_CALL_ID
        _REWARD_CALL_ID += 1
        if reward_metrics_fh is not None:
            try:
                def _mean(xs):
                    return float(sum(xs) / max(1, len(xs)))

                def _mean_opt(xs):
                    vals = [x for x in xs if x is not None]
                    return float(sum(vals) / max(1, len(vals))) if vals else None

                rec = {
                    "time": time.time(),
                    "call_id": int(_REWARD_CALL_ID),
                    "n": int(len(rewards)),
                    "reward_mean": _mean(rewards),
                    "dense_mean": _mean(dense_terms),
                    "dense_raw_mean": _mean(dense_raw_terms),
                    "terminal_mean": _mean(terminal_terms),
                    "embed_dense_mean": _mean(embed_terms),
                    "quality_mean": _mean(quality_terms),
                    "score_with_mean": _mean(score_with_terms),
                    "score_base_mean": _mean_opt(score_base_terms),
                    "terminal_mode": str(args.terminal_mode),
                    "terminal_relative": bool(args.terminal_relative),
                    "terminal_baseline": str(args.terminal_baseline),
                    "dense_weight": float(args.dense_weight),
                    "terminal_weight": float(args.terminal_weight),
                    "terminal_runtime_penalty": float(args.terminal_runtime_penalty),
                    "dense_no_terminal_scale": float(args.dense_no_terminal_scale),
                    "no_terminal_signal_rows": int(no_terminal_signal_rows),
                    "skip_baseline_perfect": bool(args.skip_baseline_perfect),
                    "baseline_perfect_threshold": float(args.baseline_perfect_threshold),
                    "skipped_baseline_perfect_rows": int(skipped_baseline_perfect_rows),
                    "quality_reward": str(args.quality_reward),
                }
                # Keep these compact (single reward batch only).
                if with_err_counts:
                    rec["with_error_codes"] = with_err_counts
                if base_err_counts:
                    rec["base_error_codes"] = base_err_counts
                if with_err_msgs:
                    rec["with_error_messages"] = dict(sorted(with_err_msgs.items(), key=lambda kv: -kv[1])[:3])
                if base_err_msgs:
                    rec["base_error_messages"] = dict(sorted(base_err_msgs.items(), key=lambda kv: -kv[1])[:3])
                if batch_qids:
                    rec["question_ids"] = batch_qids[:8]
                reward_metrics_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

        # If terminal is always zero, GRPO will optimize only dense proxies and can easily regress.
        global _WARNED_TERMINAL_DEGENERATE, _DEGENERATE_TERMINAL_CALLS
        if not _WARNED_TERMINAL_DEGENERATE:
            try:
                if len(terminal_terms) > 0 and all(abs(t) < 1e-9 for t in terminal_terms) and all(abs(s) < 1e-9 for s in score_with_terms):
                    _WARNED_TERMINAL_DEGENERATE = True
                    print(
                        "[warn] Terminal reward appears degenerate (score_with all zero in a reward batch). "
                        "Check that drafts pass some tests and that `--terminal-baseline` is set sensibly (recommended: draft). "
                        f"with_error_codes={with_err_counts} base_error_codes={base_err_counts}"
                    )
            except Exception:
                pass
        try:
            is_degenerate_call = (
                len(terminal_terms) > 0
                and all(abs(t) < 1e-9 for t in terminal_terms)
                and all(abs(s) < 1e-9 for s in score_with_terms)
                and str(args.terminal_mode) == "coder"
                and float(args.terminal_weight) > 0.0
            )
            if is_degenerate_call:
                _DEGENERATE_TERMINAL_CALLS += 1
            else:
                _DEGENERATE_TERMINAL_CALLS = 0
            if (
                bool(args.abort_on_degenerate_terminal)
                and _DEGENERATE_TERMINAL_CALLS >= int(args.degenerate_terminal_max_calls)
            ):
                raise RuntimeError(
                    "Aborting: terminal reward remained fully degenerate for "
                    f"{_DEGENERATE_TERMINAL_CALLS} consecutive reward calls. "
                    "Likely causes: coder/test execution mismatch or all-zero test signal."
                )
        except Exception:
            raise
        return rewards

    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=not args.bf16,
        max_prompt_length=args.max_prompt_tokens,
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        generation_batch_size=max(1, int(args.generation_batch_size)),
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        report_to=[],
        save_strategy="no",  # no intermediate checkpoints
        seed=args.seed,
    )
    if args.torch_empty_cache_steps and int(args.torch_empty_cache_steps) > 0:
        grpo_args.torch_empty_cache_steps = int(args.torch_empty_cache_steps)
    grpo_args.ddp_find_unused_parameters = False

    checkpoint_cb: Optional[_PeriodicCheckpointCallback] = None
    checkpoint_hooks_enabled = bool(args.enable_checkpoints) or bool(args.save_best_checkpoint)
    if checkpoint_hooks_enabled and _rank0():
        print(
            "[save] Checkpoint hooks enabled: "
            f"periodic_every={int(args.checkpoint_every_steps) if args.enable_checkpoints else 0} "
            f"keep_last={int(args.checkpoint_keep_last)} "
            f"save_best={bool(args.save_best_checkpoint)} metric={args.best_checkpoint_metric}"
        )
    if checkpoint_hooks_enabled:
        checkpoint_cb = _PeriodicCheckpointCallback(
            output_dir=Path(args.output_dir),
            processing_class=processing_class,
            base_model=str(args.model_name_or_path),
            peft_mode=str(args.peft),
            save_every_steps=(int(args.checkpoint_every_steps) if args.enable_checkpoints else 0),
            keep_last=int(args.checkpoint_keep_last),
            save_best=bool(args.save_best_checkpoint),
            best_metric=str(args.best_checkpoint_metric),
            greater_is_better=bool(args.best_checkpoint_greater_is_better),
            best_min_steps=int(args.best_checkpoint_min_steps),
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=processing_class,
        callbacks=[_ConsoleTrainLogCallback()],
        peft_config=peft_config,
    )
    if metrics_cb is not None:
        try:
            trainer.add_callback(metrics_cb)
        except Exception:
            pass
    if checkpoint_cb is not None:
        try:
            checkpoint_cb.bind_trainer(trainer)
            trainer.add_callback(checkpoint_cb)
        except Exception:
            pass

    trainer.train()
    if not args.no_save_final:
        trainer.save_model(args.output_dir)
        processing_class.save_pretrained(args.output_dir)
        (Path(args.output_dir) / "base_model.json").write_text(
            json.dumps({"base_model": args.model_name_or_path, "peft": args.peft, "script": Path(__file__).name}, indent=2),
            encoding="utf-8",
        )
        print(f"[save] Saved final model/tokenizer to {Path(args.output_dir).resolve()}")
    else:
        print("[save] Skipped saving final model (--no-save-final).")

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
    if reward_metrics_fh is not None:
        try:
            reward_metrics_fh.flush()
            reward_metrics_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
