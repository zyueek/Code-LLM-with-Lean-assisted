#!/usr/bin/env python3
"""
ITSSM (Qwen3 + Robust Reflexion) with a *post-trained* prover.

This script keeps the original ITSSM pipeline logic from:
  `fdg_approach_qwen3_inter_refined_reflex_robust.py`

but lets you swap the prover backend:
  - `vllm`: use a running prover server (original behavior)
  - `hf`: load a local HuggingFace checkpoint / PEFT adapter directory (your RL-trained prover)

Typical usage (HF prover = RL-trained adapter dir):
  python fdg_approach_qwen3_inter_posttrained.py \
    --prover-backend hf \
    --prover-model-name-or-path rl_deepseek_prover_out \
    --prover-base-model-name-or-path deepseek-ai/DeepSeek-Prover-V2-7B \
    --prover-load-in-4bit --prover-device cuda:0 \
    --qwen-base-url http://localhost:1234/v1 \
    --version v6 --max-problems 50 --reflections 1 --eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "EMPTY")

# Avoid torch.compile / inductor subprocess compilation in restricted environments.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

# Add parent directory to path to import vllm_client (matches original script behavior).
sys.path.insert(0, str(Path(__file__).parent.parent))
from vllm_client import VLLMClient  # noqa: E402

# Reuse the exact coder + pipeline implementation from the original script.
from fdg_approach_qwen3_inter_refined_reflex_robust import (  # noqa: E402
    ITSSMApproachGenerator,
    ITSSMResult,
)

def _extract_tag(text: str, tag: str) -> str:
    import re

    m = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "").strip()


def _strip_tag(text: str, tag: str) -> str:
    import re

    return re.sub(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _strip_final_code(text: str) -> str:
    # Remove RL-trained `<final_code>...</final_code>` block if it appears.
    text = _strip_tag(text, "final_code")
    # Also remove fenced code blocks (best-effort) if the model incorrectly emits code.
    import re

    text = re.sub(r"```(?:python)?\s*\n.*?\n```", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return text


def _as_bullets(block: str) -> str:
    lines = [ln.strip() for ln in (block or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    # Keep user-provided bullets if present; else bulletize lines.
    if any(ln.startswith(("-", "*")) for ln in lines):
        return "\n".join(lines)
    return "\n".join([f"- {ln}" for ln in lines])


def _count_bullets(text: str) -> int:
    return sum(1 for ln in (text or "").splitlines() if ln.strip().startswith(("-", "*")))


def _looks_overconfident(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    needles = [
        "no logical gap",
        "no gaps",
        "no issue",
        "no issues",
        "no bug",
        "no bugs",
        "already correct",
        "fully correct",
        "correct implementation",
        "correctly implement",
        "correctly implements",
        "works as expected",
        "is sound",
    ]
    return any(n in t for n in needles)


class VLLMCoderClient:
    """OpenAI-compatible client for local vLLM coder endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: Optional[int],
        context_margin_tokens: int,
        min_completion_tokens: int,
    ):
        try:
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: `openai`") from exc

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        self.model_name = model_name
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.seed = int(seed) if seed is not None else None
        self.context_margin_tokens = max(0, int(context_margin_tokens))
        self.min_completion_tokens = max(0, int(min_completion_tokens))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        import re

        t = self.temperature if temperature is None else float(temperature)
        tp = self.top_p if top_p is None else float(top_p)
        tk = self.top_k if top_k is None else int(top_k)
        rp = self.repetition_penalty if repetition_penalty is None else float(repetition_penalty)

        cur_max = max(1, int(max_tokens))
        last_exc: Optional[BaseException] = None
        for _attempt in range(3):
            try:
                extra_body = {"top_k": tk, "repetition_penalty": rp}
                chosen_seed = self.seed if seed is None else int(seed)
                if chosen_seed is not None:
                    extra_body["seed"] = int(chosen_seed)
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=cur_max,
                    temperature=t,
                    top_p=tp,
                    stop=stop,
                    extra_body=extra_body,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if "maximum context length" not in msg or "too large" not in msg:
                    raise

                m_ctx = re.search(r"maximum context length is\s*(\d+)\s*tokens", msg)
                m_inp = re.search(r"request has\s*(\d+)\s*input tokens", msg)
                if not (m_ctx and m_inp):
                    raise
                ctx = int(m_ctx.group(1))
                inp = int(m_inp.group(1))
                allowed = ctx - inp - int(self.context_margin_tokens)
                if allowed < int(self.min_completion_tokens):
                    raise RuntimeError(
                        f"Coder prompt too long: context={ctx} input={inp} margin={self.context_margin_tokens} "
                        f"min_completion={self.min_completion_tokens}."
                    ) from exc
                new_max = min(cur_max - 1, allowed)
                if new_max >= cur_max:
                    raise
                cur_max = int(new_max)
                continue
        assert last_exc is not None
        raise last_exc

    def test_connection(self) -> bool:
        try:
            resp = self.generate("Say 'ok'.", max_tokens=8, temperature=0.0)
            if resp:
                print(f"✅ vLLM coder test OK: {resp[:80]!r}")
                return True
            print("⚠️ vLLM coder reachable but returned empty output")
            return False
        except Exception as e:
            print(f"⚠️ vLLM coder not usable: {e}")
            return False


class _PublicTestExecutor:
    def __init__(self, timeout_s: int):
        from train_deepseek_prover_rl import LiveCodeBenchExecutor

        self._exec = LiveCodeBenchExecutor(timeout_s=int(timeout_s))

    @staticmethod
    def _score(tr) -> float:
        # Prefer pass/fail, else fraction of boolean results.
        if getattr(tr, "passed", False):
            return 1.0
        details = getattr(tr, "details", None) or {}
        res = details.get("results")
        if isinstance(res, list) and res:
            n = sum(1 for x in res if x is True)
            return float(n) / float(len(res))
        return 0.0

    def score(self, *, task: Dict[str, Any], solution_code: str) -> float:
        tr = self._exec.run_tests(task=task, solution_code=solution_code)
        return float(self._score(tr))
class PosttrainedITSSMApproachGenerator(ITSSMApproachGenerator):
    """
    ITSSM generator that is robust to RL-trained prover outputs.

    Key change vs the original:
    - The prover prompts are adapted for a model trained to emit tags (`<subgoal>`, `<gap_analysis>`),
      and the returned text is normalized back into the original expected sections for the coder.
    """

    def __init__(
        self,
        *args,
        prover_format: str = "auto",
        stage2_mode: str = "auto",
        prover_redteam_retry: bool = True,
        prover_redteam_max_retries: int = 1,
        draft_temperature: Optional[float] = None,
        revision_temperature: Optional[float] = None,
        draft_max_tokens: Optional[int] = None,
        revision_max_tokens: Optional[int] = None,
        draft_samples: int = 1,
        revision_samples: int = 1,
        reflections_when_public_perfect: bool = False,
        self_test_public: bool = False,
        self_test_timeout_s: int = 6,
        trust_public_tests: bool = True,
        self_test_public_mode: str = "select",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._prover_format = prover_format  # auto|original|tags
        self._stage2_mode = stage2_mode  # auto|split|combined
        self._prover_redteam_retry = bool(prover_redteam_retry)
        self._prover_redteam_max_retries = max(0, int(prover_redteam_max_retries))
        self._draft_temperature = (float(draft_temperature) if draft_temperature is not None else None)
        self._revision_temperature = (float(revision_temperature) if revision_temperature is not None else None)
        self._draft_max_tokens = (int(draft_max_tokens) if draft_max_tokens is not None else None)
        self._revision_max_tokens = (int(revision_max_tokens) if revision_max_tokens is not None else None)
        self._draft_samples = max(1, int(draft_samples))
        self._revision_samples = max(1, int(revision_samples))
        self._reflections_when_public_perfect = bool(reflections_when_public_perfect)
        self._public_tester = _PublicTestExecutor(int(self_test_timeout_s)) if bool(self_test_public) else None
        self._trust_public_tests = bool(trust_public_tests)
        self._self_test_public_mode = str(self_test_public_mode or "select").strip().lower()
        if self._self_test_public_mode not in {"select", "guard", "hint"}:
            raise ValueError(f"Invalid self_test_public_mode={self._self_test_public_mode!r}; must be select|guard|hint")
        self._redteam_hint_fail: Optional[bool] = None

    def stage1_initial(self, problem: Dict[str, Any], seed_override: Optional[int] = None) -> str:
        content = problem.get("question_content", "")
        starter_code = problem.get("starter_code", "")
        function_signature = ""
        if starter_code:
            for line in starter_code.split("\n"):
                t = line.strip()
                if t.startswith("def ") or t.startswith("class "):
                    function_signature = t
                    break
        prompt = self._build_qwen_initial_prompt(function_signature, content, starter_code)
        max_tokens = int(self.coder_max_tokens)
        if self._draft_max_tokens is not None:
            max_tokens = max(1, min(max_tokens, int(self._draft_max_tokens)))
        draft = self.coder.generate(prompt, max_tokens=max_tokens, temperature=self._draft_temperature, seed=seed_override)
        draft = self._clean_generated_code(draft)
        print(f"✅ Draft length: {len(draft)} chars")
        return draft

    def stage3_revision(
        self, problem: Dict[str, Any], current_code: str, combined_analysis: str, seed_override: Optional[int] = None
    ) -> str:
        content = problem.get("question_content", "")
        starter_code = problem.get("starter_code", "")
        function_signature = ""
        if starter_code:
            for line in starter_code.split("\n"):
                t = line.strip()
                if t.startswith("def ") or t.startswith("class "):
                    function_signature = t
                    break
        prompt = self._build_qwen_revision_prompt(
            function_signature=function_signature,
            content=content,
            initial_code=current_code,
            type_analysis=combined_analysis,
            starter_code=starter_code,
        )
        max_tokens = int(self.coder_max_tokens)
        if self._revision_max_tokens is not None:
            max_tokens = max(1, min(max_tokens, int(self._revision_max_tokens)))
        revised = self.coder.generate(
            prompt, max_tokens=max_tokens, temperature=self._revision_temperature, seed=seed_override
        )
        revised = self._clean_generated_code(revised)
        revised = self._strip_main_block(revised)
        print(f"✅ Revised length: {len(revised)} chars")
        return revised

    def _should_use_tags(self) -> bool:
        if self._prover_format == "tags":
            return True
        if self._prover_format == "original":
            return False
        # auto: HF prover tends to be RL-tuned in this script.
        return getattr(self.prover, "vllm_url", "").startswith("hf://")

    def _should_use_combined_stage2(self) -> bool:
        if self._stage2_mode == "combined":
            return True
        if self._stage2_mode == "split":
            return False
        # auto: if we're using tags, prefer combined stage2 to match `rl_training_prover_process.py`.
        return self._should_use_tags()

    def stage2_analysis_and_checklist(self, problem: Dict[str, Any], code: str) -> str:
        # When using an RL-trained prover, prefer a single stage-2 call that returns:
        #   <subgoal>...</subgoal><gap_analysis>...</gap_analysis><checklist>...</checklist>
        # This matches `rl_training_prover_process.py` and reduces mismatch vs training.
        if self._should_use_tags() and self._should_use_combined_stage2():
            content = problem.get("question_content", "")
            prompt = f"""
You are DeepSeek Prover. Do NOT write any code.
Given the problem description and the candidate implementation, identify key subgoals, a concise gap analysis,
and a robustness checklist for a coder to fix the program.

Important: treat the candidate implementation as potentially wrong. Do NOT claim it is correct or that there are no gaps.

Problem Description:
{content}

Candidate Implementation (Python):
```python
{code}
```

Return ONLY these tags:
<subgoal>one subgoal per line</subgoal>
<gap_analysis>bullet list</gap_analysis>
<checklist>bullet list</checklist>
"""
            raw = self.prover.generate(prompt, max_tokens=1400, temperature=0.0)

            def _parse(payload: str):
                payload = _strip_final_code(payload)
                return (
                    _extract_tag(payload, "subgoal"),
                    _extract_tag(payload, "gap_analysis"),
                    _extract_tag(payload, "checklist"),
                )

            subgoals, gap, checklist = _parse(raw)

            # Red-team retry if the model outputs a generic "looks correct" gap analysis.
            retries = 0
            while (
                self._prover_redteam_retry
                and retries < self._prover_redteam_max_retries
                and (self._redteam_hint_fail is not False)
                and (_looks_overconfident(gap) or _count_bullets(_as_bullets(gap)) < 2)
            ):
                retries += 1
                redteam = f"""
You are DeepSeek Prover. Do NOT write any code.
Assume the candidate implementation is wrong unless proven otherwise.
Find concrete risks and failure modes that a coder should address:
- at least 3 suspected logical bugs or spec misreads (with one minimal counterexample idea)
- at least 3 edge cases
- at least 2 Python performance pitfalls (e.g., avoid pow/modinv inside hot loops, quadratic behavior)

Problem Description:
{content}

Candidate Implementation (Python):
```python
{code}
```

Return ONLY these tags:
<subgoal>one subgoal per line</subgoal>
<gap_analysis>bullet list</gap_analysis>
<checklist>bullet list</checklist>
""".strip()
                raw2 = self.prover.generate(redteam, max_tokens=1400, temperature=0.0)
                sub2, gap2, chk2 = _parse(raw2)
                # Prefer the retry if it yields a more actionable gap analysis.
                if _count_bullets(_as_bullets(gap2)) > _count_bullets(_as_bullets(gap)):
                    subgoals, gap, checklist = sub2, gap2, chk2
                else:
                    break

            analysis = "1. Preconditions and postconditions\n- (omitted)\n\n2. Key invariants / subgoals\n"
            analysis += _as_bullets(subgoals) or "- (none)\n"
            analysis += "\n\n3. Concise gap analysis\n"
            analysis += _as_bullets(gap) or "- (none)\n"
            combined = analysis.strip() + "\n\nRobustness Checklist:\n" + (_as_bullets(checklist) or "- (none)\n").strip()
            return combined.strip()

        return super().stage2_analysis_and_checklist(problem, code)

    # ---------------- Prover helpers (restricted input) ----------------
    def _prover_type_analysis(self, problem, code: str) -> str:
        content = problem.get("question_content", "")

        if self._should_use_tags():
            prompt = f"""
You are DeepSeek Prover. Do NOT write any code.
Given the problem description and the candidate implementation, identify key subgoals and a concise gap analysis.

Problem Description:
{content}

Candidate Implementation (Python):
```python
{code}
```

Return ONLY:
<subgoal>one subgoal per line</subgoal>
<gap_analysis>bullet list</gap_analysis>
"""
            raw = self.prover.generate(prompt, max_tokens=1024, temperature=0.0)
            raw = _strip_final_code(raw)
            subgoals = _extract_tag(raw, "subgoal")
            gap = _extract_tag(raw, "gap_analysis")
            # Normalize back into the original numbered sections expected by the coder prompt.
            out = "1. Preconditions and postconditions\n- (omitted)\n\n2. Key invariants / subgoals\n"
            out += _as_bullets(subgoals) or "- (none)\n"
            out += "\n\n3. Concise gap analysis\n"
            out += _as_bullets(gap) or "- (none)\n"
            return out.strip()

        # Fallback to the original style prompt (for vLLM base prover).
        prompt = f"""
You are a type theorist analyzing a Python function. Do not write code. Infer implicit contracts and subgoals.

Problem Description:
{content}

Candidate Implementation (Python):
```python
{code}
```

Produce:
1. Preconditions and postconditions
2. Key invariants / subgoals
3. Concise gap analysis
"""
        raw = self.prover.generate(prompt, max_tokens=4096, temperature=0.0)
        return _strip_final_code(raw)

    def _prover_robustness_checklist(self, problem, code: str) -> str:
        content = problem.get("question_content", "")

        if self._should_use_tags():
            prompt = f"""
You are a competitive programming reviewer. Do NOT write code.
Given the problem description and the current Python implementation, return a robustness checklist.

Problem Description:
{content}

Current Implementation (Python):
```python
{code}
```

Return ONLY:
<checklist>bullet points only</checklist>
"""
            raw = self.prover.generate(prompt, max_tokens=600, temperature=0.0)
            raw = _strip_final_code(raw)
            block = _extract_tag(raw, "checklist") or raw
            return _as_bullets(block).strip()

        prompt = f"""
You are a competitive programming reviewer. Do not write code.
Given the problem description and the current Python implementation, list a concise robustness checklist covering:
- Output format strictness (line count, whitespace, ordering, tie-breaks)
- Edge cases (empty, single-element, extremes, duplicates, negatives/zeros)
- Algorithmic complexity targets and typical optimizations
- Data structure choices and integer-only arithmetic when appropriate

Problem Description:
{content}

Current Implementation (Python):
```python
{code}
```

Return bullet points only.
"""
        raw = self.prover.generate(prompt, max_tokens=512, temperature=0.0)
        return _strip_final_code(raw)

    def _prover_feedback(self, problem, code: str) -> str:
        content = problem.get("question_content", "")

        if self._should_use_tags():
            prompt = f"""
You are reviewing the Python implementation against the problem description. Do NOT write code.
Return concise, actionable revision feedback focusing on: edge cases, incorrect logic, invariants, complexity, and output format.

Problem Description:
{content}

Current Implementation (Python):
```python
{code}
```

Return ONLY:
<feedback>bullets only</feedback>
"""
            raw = self.prover.generate(prompt, max_tokens=600, temperature=0.0)

            def _parse(payload: str) -> str:
                payload = _strip_final_code(payload)
                block = _extract_tag(payload, "feedback") or payload
                return _as_bullets(block).strip()

            fb = _parse(raw)
            retries = 0
            while (
                self._prover_redteam_retry
                and retries < self._prover_redteam_max_retries
                and (self._redteam_hint_fail is not False)
                and (_looks_overconfident(fb) or _count_bullets(fb) < 3)
            ):
                retries += 1
                redteam = f"""
You are reviewing the Python implementation against the problem description. Do NOT write code.
Assume the solution is wrong unless proven otherwise. Provide at least 5 bullets that are *specific* and actionable.
Include:
- one plausible counterexample / failing scenario idea
- one Python performance pitfall to check
- one strict I/O / formatting pitfall to check

Problem Description:
{content}

Current Implementation (Python):
```python
{code}
```

Return ONLY:
<feedback>bullets only</feedback>
""".strip()
                fb2 = _parse(self.prover.generate(redteam, max_tokens=600, temperature=0.0))
                if _count_bullets(fb2) > _count_bullets(fb):
                    fb = fb2
                else:
                    break
            return fb

        prompt = f"""
You are reviewing the Python implementation against the problem description. Provide a concise, actionable feedback list for one more revision. Do NOT write any code.
Focus on: missing edge cases, incorrect logic, invariants, complexity, and output formatting strictness.

Format strictly as bullets.

Problem Description:
{content}

Current Implementation (Python):
```python
{code}
```
"""
        raw = self.prover.generate(prompt, max_tokens=600, temperature=0.0)
        return _strip_final_code(raw)

    def load_livecodebench_dataset(self, version_tag: str = "v6") -> List[Dict[str, Any]]:
        # Prefer offline cached dataset loader to avoid network dependency.
        try:
            from train_deepseek_prover_rl import _load_tasks_from_livecodebench

            tasks = _load_tasks_from_livecodebench([version_tag], limit=None, start_index=0, end_index=None)
            for t in tasks:
                qid = str(t.get("question_id") or "")
                if "/" in qid:
                    t["question_id"] = qid.split("/", 1)[1]
            print(f"Loaded LiveCodeBench-Lite dataset ({version_tag}) from local cache: {len(tasks)} problems")
            return tasks
        except Exception as e:
            print(f"⚠️ Offline dataset loader failed: {e}")
            return super().load_livecodebench_dataset(version_tag)

    def generate_one(self, problem: Dict[str, Any], index: int):
        # Optional public-test selection to mitigate regressions when a critic breaks an otherwise-correct draft.
        task_id = self.get_task_id(problem, index)
        print(f"\n🎯 ITSSM Robust Reflex for {task_id}")

        def _hint_from_score(score: Optional[float]) -> Optional[bool]:
            if score is None:
                return None
            if score < 1.0:
                return True
            # If public tests pass perfectly: this is not proof of correctness on hidden tests.
            # If `trust_public_tests` is enabled, we treat it as "likely correct" and skip red-team retries.
            # Otherwise we keep the hint unknown and allow red-team retries on overconfident critiques.
            return False if self._trust_public_tests else None

        def _score_public(code: str) -> Optional[float]:
            if self._public_tester is None:
                return None
            try:
                return float(self._public_tester.score(task=problem, solution_code=code))
            except Exception:
                return None

        best_code = ""
        best_score: float = -1.0

        def _update_best(code: str, score: Optional[float]) -> None:
            nonlocal best_code, best_score
            if self._public_tester is None or self._self_test_public_mode == "hint" or score is None:
                return
            s = float(score)
            if s > best_score:
                best_score = s
                best_code = code
                return
            if s == best_score and self._self_test_public_mode == "guard":
                best_score = s
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
        for di in range(int(actual_drafts)):
            seed_i = (base_seed + di) if base_seed is not None else None
            cand = self.stage1_initial(problem, seed_override=seed_i)
            cand_score = _score_public(cand)
            if not draft:
                draft = cand
                draft_score = cand_score
            else:
                if cand_score is not None and draft_score is not None:
                    if float(cand_score) > float(draft_score) or (
                        float(cand_score) == float(draft_score) and self._self_test_public_mode == "guard"
                    ):
                        draft = cand
                        draft_score = cand_score
            _update_best(cand, cand_score)
        self._redteam_hint_fail = _hint_from_score(draft_score)

        combined = self.stage2_analysis_and_checklist(problem, draft)

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
        for ri in range(int(actual_revs)):
            seed_i = (base_seed + 1000 + ri) if base_seed is not None else None
            cand = self.stage3_revision(problem, draft, combined, seed_override=seed_i)
            cand_score = _score_public(cand)
            if not revised:
                revised = cand
                revised_score = cand_score
            else:
                if cand_score is not None and revised_score is not None:
                    if float(cand_score) > float(revised_score) or (
                        float(cand_score) == float(revised_score) and self._self_test_public_mode == "guard"
                    ):
                        revised = cand
                        revised_score = cand_score
            _update_best(cand, cand_score)

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

            new_revised = ""
            new_score: Optional[float] = None
            for ri in range(int(actual_revs)):
                seed_i = (base_seed + 2000 + r * 50 + ri) if base_seed is not None else None
                cand = self.stage3_revision(problem, revised, combined2, seed_override=seed_i)
                cand_score = _score_public(cand)
                if not new_revised:
                    new_revised = cand
                    new_score = cand_score
                else:
                    if cand_score is not None and new_score is not None:
                        if float(cand_score) > float(new_score) or (
                            float(cand_score) == float(new_score) and self._self_test_public_mode == "guard"
                        ):
                            new_revised = cand
                            new_score = cand_score
                _update_best(cand, cand_score)

            if new_revised.strip() == revised.strip():
                print("🔁 No effective change; stop.")
                combined = combined2
                break
            revised = new_revised
            revised_score = new_score
            combined = combined2

        # If enabled, pick the best code by public tests, but keep the final `type_analysis` for debugging.
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


class HFProverClient:
    """
    Local HuggingFace/Transformers prover client compatible with ITSSMApproachGenerator.

    - If `model_name_or_path` is a PEFT adapter directory, pass `base_model_name_or_path`
      (or ensure `base_model.json` exists in the adapter dir).
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        base_model_name_or_path: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        bf16: bool = False,
        trust_remote_code: bool = True,
        merge_lora: bool = False,
    ):
        self.model_name = model_name_or_path
        self.vllm_url = f"hf://{model_name_or_path}"
        self._device_str = device
        self._load_in_4bit = bool(load_in_4bit)
        self._bf16 = bool(bf16)
        self._trust_remote_code = bool(trust_remote_code)
        self._merge_lora = bool(merge_lora)

        self._tokenizer = None
        self._model = None

        self._base_model_name_or_path = base_model_name_or_path or self._load_base_model_hint(Path(model_name_or_path))
        self._load()

    @staticmethod
    def _is_peft_adapter_dir(path: Path) -> bool:
        return (
            path.exists()
            and (
                (path / "adapter_config.json").exists()
                or (path / "adapter_model.safetensors").exists()
                or (path / "adapter_model.bin").exists()
            )
        )

    @staticmethod
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

    def _resolve_device(self) -> str:
        if self._device_str and self._device_str != "auto":
            return self._device_str
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()
        is_adapter = self._is_peft_adapter_dir(Path(self.model_name))

        dtype = torch.bfloat16 if self._bf16 else torch.float16

        tok_source = self.model_name
        if is_adapter and not (Path(self.model_name) / "tokenizer_config.json").exists():
            tok_source = self._base_model_name_or_path or self.model_name

        tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True, trust_remote_code=self._trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        device_map = None
        load_in_4bit = self._load_in_4bit
        if load_in_4bit:
            if device.startswith("cuda:"):
                idx = int(device.split(":")[1])
                device_map = {"": idx}
                torch.cuda.set_device(idx)
            elif device == "cuda":
                device_map = {"": 0}
                torch.cuda.set_device(0)
            else:
                load_in_4bit = False
                print("⚠️ --prover-load-in-4bit requested but device is CPU; falling back to non-4bit.")

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=dtype,
            )

        if is_adapter:
            if not self._base_model_name_or_path:
                raise ValueError(
                    "HF prover adapter directory detected but base model is unknown. "
                    "Pass --prover-base-model-name-or-path or include base_model.json in the adapter dir."
                )
            base = AutoModelForCausalLM.from_pretrained(
                self._base_model_name_or_path,
                trust_remote_code=self._trust_remote_code,
                torch_dtype=None if load_in_4bit else dtype,
                quantization_config=quantization_config,
                device_map=device_map,
            )
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError("Loading a PEFT adapter requires `peft` to be installed.") from exc

            model = PeftModel.from_pretrained(base, self.model_name)
            if self._merge_lora:
                try:
                    model = model.merge_and_unload()
                except Exception:
                    pass
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self._trust_remote_code,
                torch_dtype=None if load_in_4bit else dtype,
                quantization_config=quantization_config,
                device_map=device_map,
            )

        if not load_in_4bit:
            model = model.to(device)

        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("HFProverClient not initialized.")

        device = self._resolve_device()
        t = 0.0 if temperature is None else float(temperature)
        tp = 1.0 if top_p is None else float(top_p)
        tk = None if top_k is None else int(top_k)
        rp = 1.0 if repetition_penalty is None else float(repetition_penalty)

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            gen_kwargs = dict(
                do_sample=t > 0.0,
                temperature=max(1e-5, t) if t > 0.0 else 1.0,
                top_p=tp,
                max_new_tokens=int(max_tokens),
                repetition_penalty=rp,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            if t > 0.0 and tk is not None:
                gen_kwargs["top_k"] = tk
            gen = self._model.generate(**inputs, **gen_kwargs)

        out_ids = gen[0][prompt_len:]
        text = self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()

        if stop:
            for s in stop:
                if not s:
                    continue
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx].strip()
        return text

    def test_connection(self) -> bool:
        try:
            resp = self.generate("Say 'ok'.", max_tokens=8, temperature=0.0)
            if resp:
                print(f"✅ HF prover test OK: {resp[:80]!r}")
                return True
            print("⚠️ HF prover loaded but returned empty output")
            return False
        except Exception as e:
            print(f"⚠️ HF prover not usable: {e}")
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="ITSSM (Qwen-style prompts) with Robust Reflexion + post-trained prover")
    parser.add_argument("--prover-backend", type=str, choices=["vllm", "hf"], default="vllm")
    parser.add_argument(
        "--prover-url",
        type=str,
        default=os.getenv("PROVER_BASE_URL", "http://localhost:5678/"),
        help="(vllm) Prover (DeepSeek Prover) OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--prover-model",
        type=str,
        default=os.getenv("PROVER_MODEL", "deepseek-ai/DeepSeek-Prover-V2-7B"),
        help="(vllm) Served model name/id to request from the prover server (supports LoRA-served names)",
    )
    parser.add_argument("--prover-model-name-or-path", type=str, default=None, help="(hf) Prover model path or adapter dir")
    parser.add_argument("--prover-base-model-name-or-path", type=str, default=None, help="(hf, adapter only) Base model id/path")
    parser.add_argument("--prover-device", type=str, default="auto", help="(hf) auto|cpu|cuda|cuda:0|cuda:1")
    parser.add_argument("--prover-load-in-4bit", action="store_true", help="(hf) Load base model in 4-bit (CUDA only)")
    parser.add_argument("--prover-bf16", action="store_true", help="(hf) Use bf16 (if supported) for non-4bit inference")
    parser.add_argument("--prover-merge-lora", action="store_true", help="(hf, adapter) Merge LoRA into base model for inference")
    parser.add_argument(
        "--prover-format",
        type=str,
        choices=["auto", "original", "tags"],
        default="auto",
        help="How to prompt/parse prover outputs. Use `tags` to match GRPO-trained prover output format.",
    )
    parser.add_argument(
        "--stage2-mode",
        type=str,
        choices=["auto", "split", "combined"],
        default="auto",
        help="How Stage-2 prover is called. `combined` matches rl_training_prover_process.py (subgoal+gap+checklist in one call).",
    )

    parser.add_argument("--output-dir", type=str, default="./itssm_approach_results")
    parser.add_argument("--version", type=str, default="v6")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test-connections", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=175)
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    parser.add_argument("--qwen-base-url", type=str, default=None, help="Override vLLM base URL (default http://localhost:1234/v1)")

    parser.add_argument("--coder-max-tokens", type=int, default=15000, help="Max tokens for coder generation in stage-1/stage-3.")
    parser.add_argument(
        "--coder-context-margin-tokens",
        type=int,
        default=256,
        help="Reserved token headroom when recovering from vLLM context overflow.",
    )
    parser.add_argument(
        "--coder-min-completion-tokens",
        type=int,
        default=256,
        help="Fail fast if context overflow leaves fewer than this many completion tokens.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for vLLM sampling (improves reproducibility).")
    parser.add_argument("--run-timestamp", type=int, default=None, help="Fixed timestamp for reproducible output filenames.")

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--top-k", dest="top_k", type=int, default=20)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.05)
    parser.add_argument("--draft-temperature", type=float, default=None, help="Optional override temperature for stage-1 draft generation.")
    parser.add_argument(
        "--revision-temperature",
        type=float,
        default=None,
        help="Optional override temperature for stage-3 revision generation (recommended lower than draft, e.g. 0.2).",
    )
    parser.add_argument("--draft-max-tokens", type=int, default=None, help="Optional override max_tokens for stage-1 (<= --coder-max-tokens).")
    parser.add_argument("--revision-max-tokens", type=int, default=None, help="Optional override max_tokens for stage-3 (<= --coder-max-tokens).")
    parser.add_argument("--reflections", type=int, default=1)
    parser.add_argument("--draft-samples", type=int, default=1, help="Sample N independent stage-1 drafts (selected by public tests when enabled).")
    parser.add_argument(
        "--revision-samples",
        type=int,
        default=1,
        help="Sample N independent stage-3 revisions per critique (selected by public tests when enabled).",
    )
    parser.add_argument("--self-test-public", action="store_true", help="Run LiveCodeBench public tests during generation and select the best code.")
    parser.add_argument(
        "--self-test-public-mode",
        type=str,
        choices=["select", "guard", "hint"],
        default="select",
        help="How public-test results affect the final output: select=max score (ties keep earlier), guard=max score (ties prefer later), hint=do not select, only use tests as a red-team hint.",
    )
    parser.add_argument("--self-test-timeout-s", type=int, default=6, help="Per-problem public test timeout (seconds).")
    parser.add_argument(
        "--trust-public-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --self-test-public is enabled, treat a perfect public score as evidence to skip red-team retries.",
    )
    parser.add_argument(
        "--reflections-when-public-perfect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true and public tests are trusted, skip reflections when best public score is perfect.",
    )
    parser.add_argument(
        "--prover-redteam-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry prover critique/feedback with a red-team prompt when output is overconfident or too thin.",
    )
    parser.add_argument("--prover-redteam-max-retries", type=int, default=1, help="Max red-team retries per prover call.")
    parser.add_argument(
        "--require-checkpoint-best",
        action="store_true",
        help="(hf) If --prover-model-name-or-path is a run dir, require and use its checkpoint-best subdir.",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.qwen_base_url:
        os.environ["QWEN_VLLM_BASE_URL"] = args.qwen_base_url

    print("🔗 Connecting to servers...")
    if args.prover_backend == "vllm":
        prover_client = VLLMClient(base_url=args.prover_url, model_name=args.prover_model)
    else:
        if not args.prover_model_name_or_path:
            raise SystemExit("--prover-backend hf requires --prover-model-name-or-path")
        if args.require_checkpoint_best:
            base = Path(args.prover_model_name_or_path)
            cand = base / "checkpoint-best"
            if cand.exists() and cand.is_dir():
                args.prover_model_name_or_path = str(cand)
            else:
                raise SystemExit(f"--require-checkpoint-best set but not found: {cand}")
        print("📦 Loading local HF prover...")
        prover_client = HFProverClient(
            model_name_or_path=args.prover_model_name_or_path,
            base_model_name_or_path=args.prover_base_model_name_or_path,
            device=args.prover_device,
            load_in_4bit=args.prover_load_in_4bit,
            bf16=args.prover_bf16,
            merge_lora=args.prover_merge_lora,
        )

    qwen_base_url = args.qwen_base_url or os.getenv("QWEN_VLLM_BASE_URL", "http://localhost:1234/v1")
    coder_client = VLLMCoderClient(
        api_key="EMPTY",
        model_name=args.qwen_model,
        base_url=qwen_base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        context_margin_tokens=args.coder_context_margin_tokens,
        min_completion_tokens=args.coder_min_completion_tokens,
    )

    if args.test_connections:
        print("\n🧪 Testing connections...")
        prover_ok = prover_client.test_connection()
        coder_ok = coder_client.test_connection()
        if prover_ok and coder_ok:
            print("✅ Both connections successful! Ready for generation.")
        else:
            print("❌ Connection test failed!")
            if not prover_ok:
                print(f"  - Prover not accessible/usable (backend={args.prover_backend})")
            if not coder_ok:
                print("  - Qwen vLLM API not accessible")
        return

    generator = PosttrainedITSSMApproachGenerator(
        prover_client,
        coder_client,
        args.output_dir,
        reflections=args.reflections,
        coder_max_tokens=args.coder_max_tokens,
        prover_format=args.prover_format,
        stage2_mode=args.stage2_mode,
        prover_redteam_retry=bool(args.prover_redteam_retry),
        prover_redteam_max_retries=int(args.prover_redteam_max_retries),
        draft_temperature=args.draft_temperature,
        revision_temperature=args.revision_temperature,
        draft_max_tokens=args.draft_max_tokens,
        revision_max_tokens=args.revision_max_tokens,
        draft_samples=int(args.draft_samples),
        revision_samples=int(args.revision_samples),
        reflections_when_public_perfect=bool(args.reflections_when_public_perfect),
        self_test_public=bool(args.self_test_public),
        self_test_timeout_s=int(args.self_test_timeout_s),
        trust_public_tests=bool(args.trust_public_tests),
        self_test_public_mode=str(args.self_test_public_mode),
    )
    max_problems = None if args.all else args.max_problems
    results = generator.generate_batch(
        max_problems=max_problems,
        version_tag=args.version,
        start_index=args.start_index,
        end_index=args.end_index,
        run_timestamp=args.run_timestamp,
    )

    print("\n" + "=" * 70)
    print("ITSSM Robust Reflex SUMMARY")
    print("=" * 70)
    successful = len([r for r in results if not r.revised_python.startswith("# Error")])
    print(f"Problems processed: {len(results)}")
    print(f"Successful generations: {successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")

    if args.eval:
        print("\n🧪 Running LiveCodeBench evaluation...")
        import glob

        livebench_files = glob.glob(str(Path(args.output_dir) / "itssm_robust_reflex_livebench_*.json"))
        if livebench_files:
            latest_file = max(livebench_files)
            print(f"Evaluating: {latest_file}")
            try:
                eval_results, _temp_dir = generator.run_livecodebench_evaluation(latest_file, args.version)
                if eval_results:
                    results_file = latest_file.replace(".json", "_results.json")
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(eval_results, f, indent=2)
                    print(f"Evaluation results saved to: {results_file}")
                    print("✅ Evaluation completed!")
                else:
                    print("⚠️ Evaluation completed but no results file generated")
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
        else:
            print("❌ No LiveCodeBench files found to evaluate")


if __name__ == "__main__":
    main()
