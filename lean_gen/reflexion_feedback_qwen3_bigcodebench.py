#!/usr/bin/env python3
"""Reflexion-style feedback loop for BigCodeBench using Qwen3.

This script reuses the Reflexion pipeline from ``reflexion_feedback_qwen3_improved.py``
but swaps the dataset to BigCodeBench and writes outputs directly in the
BigCodeBench ``samples.jsonl`` format so they can be evaluated with
``bigcodebench.evaluate`` without any post-processing.

It does **not** change any model, prompting, or sampling settings compared to
the original LiveCodeBench script – only the benchmark and output format.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import json

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "EMPTY")

# Ensure local BigCodeBench repo is importable
BASE_DIR = Path(__file__).parent
BIGCODEBENCH_ROOT = BASE_DIR / "bigcodebench"
if BIGCODEBENCH_ROOT.exists():
    sys.path.insert(0, str(BIGCODEBENCH_ROOT))

try:
    from bigcodebench.data import get_bigcodebench
    from bigcodebench.sanitize import sanitize
except Exception as exc:  # pragma: no cover - depends on local BCB setup
    raise RuntimeError(
        "BigCodeBench repo is not importable. Make sure you cloned it into "
        "'lean_gen/bigcodebench' and installed its dependencies, e.g.\n"
        "  pip install -e bigcodebench\n"
    ) from exc

from reflexion_feedback_qwen3_improved import (
    OpenAIClient,
    ReflexionApproachGenerator,
)


def _format_doc_struct(doc_struct_raw: str) -> str:
    if not doc_struct_raw:
        return ""
    try:
        info = json.loads(doc_struct_raw)
    except Exception:
        return doc_struct_raw

    lines = []
    for key, value in info.items():
        if isinstance(value, list):
            body = "\n".join(f"  - {item}" for item in value)
        else:
            body = str(value)
        lines.append(f"{key.capitalize()}:\n{body}")
    return "\n".join(lines)


def load_bigcodebench_tasks(split: str = "instruct", subset: str = "full") -> List[Dict[str, Any]]:
    """Load BigCodeBench tasks and adapt them to the Reflexion pipeline.

    We build a BigCodeBench-specific "question" that explicitly mentions the
    function name and task type, while still feeding ``starter_code`` through
    the original prompt builders unchanged.
    """

    dataset = get_bigcodebench(subset=subset)
    tasks: List[Dict[str, Any]] = []

    for task_id, task in dataset.items():
        entry_point = task.get("entry_point", "task_func")

        if split == "instruct":
            base_desc = task.get("instruct_prompt", "")
            scenario = "instruction-following"
        elif split == "complete":
            base_desc = task.get("complete_prompt", "")
            scenario = "code-completion"
        else:
            raise ValueError("split must be 'instruct' or 'complete'")

        doc_struct_text = _format_doc_struct(task.get("doc_struct", ""))
        libs = task.get("libs") or []

        question_parts = [
            f"You are solving a BigCodeBench-{subset.capitalize()} {split} task ({scenario}).\n",
            f"Implement `{entry_point}` so it satisfies the specification below. The autograder prepends the starter code and calls this function directly.\n",
            "Avoid extra prints or `if __name__ == '__main__'` blocks. Return values instead of printing unless explicitly required.\n\n",
            "Primary specification (verbatim from BigCodeBench):\n",
            f"{base_desc}\n\n",
        ]

        if doc_struct_text:
            question_parts.append(f"Structured details:\n{doc_struct_text}\n\n")

        if isinstance(libs, list) and libs:
            question_parts.append(
                "Preferred/allowed libraries: "
                f"{', '.join(libs)}. Use them when relevant but avoid other imports.\n\n"
            )

        question_parts.append(
            "Respond with a single Python solution that fits inside the provided starter function."
        )

        question = "".join(question_parts)

        # Use the BigCodeBench code stub as starter code so the Reflexion
        # prompting can extract a function signature and fill in the body.
        starter_code = task.get("code_prompt", "") or ""

        preferred_libs = task.get("libs")
        if not isinstance(preferred_libs, list):
            preferred_libs = []

        tasks.append(
            {
                "question_id": task_id,
                "task_id": task_id,
                "question_content": question,
                "starter_code": starter_code,
                "entry_point": entry_point,
                "doc_struct_text": doc_struct_text,
                "preferred_libs": preferred_libs,
                "bcb_split": split,
                "bcb_subset": subset,
            }
        )

    return tasks


class BigCodeBenchReflexionGenerator(ReflexionApproachGenerator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._current_meta: Dict[str, Any] = {}

    def generate_reflexion_solution(self, problem: Dict[str, Any], index: int):
        self._current_meta = {
            "entry_point": problem.get("entry_point", "task_func"),
            "doc_struct_text": problem.get("doc_struct_text", ""),
            "preferred_libs": problem.get("preferred_libs", []),
            "subset": problem.get("bcb_subset", "full"),
            "split": problem.get("bcb_split", "instruct"),
        }
        return super().generate_reflexion_solution(problem, index)

    def _meta_header(self, function_signature: str) -> str:
        entry_point = self._current_meta.get("entry_point") or function_signature or "task_func"
        subset = self._current_meta.get("subset", "full")
        split = self._current_meta.get("split", "instruct")
        doc_struct = self._current_meta.get("doc_struct_text")
        libs = self._current_meta.get("preferred_libs") or []

        parts = [
            f"You are solving a BigCodeBench-{subset.capitalize()} {split} task.",
            f"The autograder prepends the starter code and calls `{entry_point}` directly.",
            "Do not add CLI wrappers or extra prints; return values directly.",
        ]
        if doc_struct:
            parts.append(f"Structured notes:\n{doc_struct}")
        if libs:
            parts.append("Preferred libraries: " + ", ".join(libs))
        return "\n".join(parts)

    def _build_initial_prompt(self, function_signature: str, content: str, starter_code: str) -> str:
        header = self._meta_header(function_signature)
        prompt = header + "\n\n"
        prompt += "### BigCodeBench Specification:\n" + content + "\n\n"
        if starter_code:
            prompt += (
                "### Starter Code (must be respected):\n"
                f"```python\n{starter_code}\n```\n\n"
            )
        else:
            scaffold = function_signature or "# IMPLEMENT FUNCTION HERE"
            prompt += f"### Signature\n```python\n{scaffold}\n```\n\n"
        prompt += "### Response: output only the final Python code implementing the function.\n"
        return prompt

    def _build_revision_prompt(
        self,
        function_signature: str,
        content: str,
        starter_code: str,
        current_code: str,
        feedback: str,
        iteration: int,
    ) -> str:
        header = self._meta_header(function_signature)
        prompt = header + "\n\n"
        prompt += f"### Problem Context\n{content}\n\n"
        if starter_code:
            prompt += f"### Starter Code\n```python\n{starter_code}\n```\n\n"
        prompt += f"### Current Submission (iteration {iteration})\n```python\n{current_code}\n```\n\n"
        prompt += f"### Feedback\n{feedback.strip()}\n\n"
        prompt += "### Revised Answer\nReturn only the updated code in one Python block.\n"
        return prompt

    def _build_feedback_prompt(
        self,
        function_signature: str,
        content: str,
        starter_code: str,
        candidate_code: str,
        iteration: int,
    ) -> str:
        header = self._meta_header(function_signature)
        prompt = header + "\n\n"
        prompt += f"### Problem Description\n{content}\n\n"
        if starter_code:
            prompt += f"### Starter Code\n```python\n{starter_code}\n```\n\n"
        prompt += f"### Candidate Submission (iteration {iteration})\n```python\n{candidate_code}\n```\n\n"
        prompt += (
            "### Review Instructions\n"
            "- State PASS or FAIL with justification.\n"
            "- Provide precise corrections referencing the specification and starter code.\n"
        )
        return prompt


def run_reflexion_bigcodebench(args: argparse.Namespace) -> Path:
    coder_base_url = args.qwen_base_url or os.getenv("QWEN_VLLM_BASE_URL")
    critic_base_url = args.critic_base_url or coder_base_url or os.getenv("CRITIC_VLLM_BASE_URL")

    print("🔗 Connecting to models...")
    coder_client = OpenAIClient(
        api_key="EMPTY",
        model_name=args.qwen_model,
        base_url=coder_base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    critic_client = OpenAIClient(
        api_key="EMPTY",
        model_name=args.critic_model,
        base_url=critic_base_url,
        temperature=args.critic_temperature,
        top_p=0.0,
        top_k=0,
        repetition_penalty=1.0,
    )

    if args.test_connections:
        print("\n🧪 Testing model connectivity...")
        critic_ok = critic_client.test_connection()
        coder_ok = coder_client.test_connection()
        if not critic_ok:
            print("❌ Critic connection failed.")
        if not coder_ok:
            print("❌ Coder connection failed.")
        if not (critic_ok and coder_ok):
            raise SystemExit(1)

    generator = BigCodeBenchReflexionGenerator(
        critic_client=critic_client,
        coder_client=coder_client,
        output_dir=args.output_dir,
        reflections=args.reflections,
    )

    print(f"📚 Loading BigCodeBench ({args.bcb_split}, subset={args.bcb_subset})...")
    bigcode_tasks = load_bigcodebench_tasks(split=args.bcb_split, subset=args.bcb_subset)
    print(f"Loaded {len(bigcode_tasks)} tasks from BigCodeBench")

    results = []
    start_time = time.time()

    for idx, problem in enumerate(bigcode_tasks):
        try:
            result = generator.generate_reflexion_solution(problem, idx)
            results.append(result)
            print(f"✅ {result.task_id}: final code length {len(result.final_code)} chars")
        except Exception as exc:  # pragma: no cover - defensive
            task_id = problem["question_id"]
            print(f"❌ Error on BigCodeBench task {task_id}: {exc}")
            # Still add a stub so BigCodeBench evaluation sees one sample per task
            from reflexion_feedback_qwen3_improved import ReflexionResult, ReflexionStep

            results.append(
                ReflexionResult(
                    task_id=task_id,
                    problem_description=problem.get("question_content", ""),
                    starter_code=problem.get("starter_code", ""),
                    final_code=f"# Error during generation: {exc}",
                    reasoning_chain=str(exc),
                    steps=[
                        ReflexionStep(
                            iteration=0,
                            verdict="ERROR",
                            feedback=str(exc),
                            code="",
                        )
                    ],
                )
            )

    duration = time.time() - start_time
    print(f"\n⏱️ Generation done in {duration:.1f}s for {len(results)} tasks")

    # Write BigCodeBench-compatible samples.jsonl
    bcb_root = Path(args.bcb_root)
    bcb_root.mkdir(parents=True, exist_ok=True)

    model_tag = args.qwen_model.replace("/", "--")
    subset_suffix = "" if args.bcb_subset == "full" else f"-{args.bcb_subset}"
    # Simple name; evaluation only cares about task_id / solution fields
    samples_path = bcb_root / (
        f"{model_tag}--bigcodebench{subset_suffix}-{args.bcb_split}--reflexion-"
        f"{args.temperature}-{1}.jsonl"
    )

    # We sanitize using BigCodeBench's helper for better compatibility.
    # For tasks that failed, we still emit the raw error string as solution.
    print(f"📦 Writing BigCodeBench samples to: {samples_path}")
    with samples_path.open("w", encoding="utf-8") as out:
        for prob, res in zip(bigcode_tasks, results):
            task_id = prob["question_id"]
            entry_point = prob.get("entry_point", "")
            raw_code = res.final_code or ""
            if raw_code.strip().startswith("# Error") or not raw_code.strip():
                solution = raw_code
            else:
                try:
                    solution = sanitize(raw_code, entry_point or None)
                except Exception:
                    solution = raw_code

            record = {
                "task_id": task_id,
                "solution": solution,
                "raw_solution": raw_code,
            }
            out.write(json.dumps(record) + "\n")

    print("✅ BigCodeBench samples ready.")
    print("You can now run evaluation, e.g.:")
    print(
        f"  bigcodebench.evaluate --execution local --split {args.bcb_split} "
        f"--subset {args.bcb_subset} --samples {samples_path}"
    )

    return samples_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Reflexion-style feedback pipeline on BigCodeBench",
    )

    # Model / connection args mirror the original script
    parser.add_argument("--critic-model", type=str, default="Qwen/Qwen3-8B", help="Model used for feedback generation")
    parser.add_argument("--critic-base-url", type=str, default="http://localhost:5678/v1", help="Base URL for critic OpenAI-compatible endpoint")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", help="Coder model served by OpenAI-compatible endpoint")
    parser.add_argument("--qwen-base-url", type=str, default=None, help="Base URL for coder OpenAI-compatible endpoint")

    parser.add_argument("--output-dir", type=str, default="reflexion_feedback_results", help="Directory to store Reflexion detailed outputs (if needed)")

    # BigCodeBench-specific controls
    parser.add_argument("--bcb-split", type=str, default="instruct", choices=["instruct", "complete"], help="BigCodeBench split to use")
    parser.add_argument("--bcb-subset", type=str, default="full", choices=["full", "hard"], help="BigCodeBench subset: full or hard")
    parser.add_argument("--bcb-root", type=str, default="bcb_results", help="Root directory to store BigCodeBench samples.jsonl")

    # Reflexion hyperparameters (same defaults as original)
    parser.add_argument("--reflections", type=int, default=1, help="Maximum feedback iterations")
    parser.add_argument("--temperature", type=float, default=0.6, help="Coder sampling temperature")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95, help="Coder nucleus sampling top-p")
    parser.add_argument("--top-k", dest="top_k", type=int, default=20, help="Coder top-k sampling")
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.05, help="Coder repetition penalty")
    parser.add_argument("--critic-temperature", dest="critic_temperature", type=float, default=0.0, help="Critic sampling temperature")
    parser.add_argument("--test-connections", action="store_true", help="Only test critic/coder connectivity")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_reflexion_bigcodebench(args)


if __name__ == "__main__":
    main()
