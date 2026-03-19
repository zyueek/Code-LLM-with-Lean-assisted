#!/usr/bin/env python3
"""
Reflexion-style feedback loop for LiveCodeBench using Qwen3 coder and optional critic.

This script mirrors the dataset/model setup from fdg_approach_qwen3_inter_refined.py while
incorporating a feedback-driven refinement cycle similar to Reflexion/TextGrad.
"""

import json
import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "EMPTY")

# Ensure shared utilities are importable when running as a module inside the repo
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import openai
except ImportError:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
    import openai

try:
    from datasets import load_dataset
except ImportError:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
    from datasets import load_dataset


class OpenAIClient:
    """Client wrapper for an OpenAI-compatible chat endpoint (e.g., vLLM)."""

    def __init__(
        self,
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        base_url: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ):
        resolved_base_url = base_url or os.getenv("QWEN_VLLM_BASE_URL", "http://localhost:1234/v1")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=resolved_base_url,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.base_url = resolved_base_url
        print(f"Initialized OpenAI-compatible client with model {self.model_name} @ {self.base_url}")

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
        try:
            print(f"🔍 Sending prompt to {self.model_name} (first 200 chars): {prompt[:200]}...")
            t = self.temperature if temperature is None else temperature
            tp = self.top_p if top_p is None else top_p
            tk = self.top_k if top_k is None else top_k
            rp = self.repetition_penalty if repetition_penalty is None else repetition_penalty

            extra_body: Dict[str, Any] = {}
            if tk is not None and tk > 0:
                extra_body["top_k"] = tk
            if rp is not None and rp != 1.0:
                extra_body["repetition_penalty"] = rp

            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": t,
                "stop": stop,
            }

            if tp is not None and tp > 0.0:
                kwargs["top_p"] = tp
            if extra_body:
                kwargs["extra_body"] = extra_body

            completion = self.client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content.strip()
            print(f"✅ {self.model_name} generated {len(text)} characters")
            return text
        except Exception as exc:
            print(f"❌ Error generating with {self.model_name}: {exc}")
            return ""

    def test_connection(self) -> bool:
        try:
            response = self.generate("def ping():\n    pass", max_tokens=8, temperature=0.0)
            if response:
                print(f"✅ Connection ok: {response[:40]}...")
                return True
        except Exception as exc:
            print(f"⚠️ Connection failed: {exc}")
        return False


@dataclass
class ReflexionStep:
    iteration: int
    verdict: str
    feedback: str
    code: str


@dataclass
class ReflexionResult:
    task_id: str
    problem_description: str
    starter_code: str
    final_code: str
    reasoning_chain: str
    steps: List[ReflexionStep] = field(default_factory=list)


class ReflexionApproachGenerator:
    """Implements a Reflexion-style feedback loop on LiveCodeBench problems."""

    def __init__(
        self,
        critic_client: OpenAIClient,
        coder_client: OpenAIClient,
        output_dir: str = "./reflexion_feedback_results",
        reflections: int = 1,
    ):
        self.critic = critic_client
        self.coder = coder_client
        self.reflections = max(0, reflections)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print("🌀 Reflexion Pipeline Initialized")
        print(f"  Critic LLM: {self.critic.model_name}")
        print(f"  Coder LLM: {self.coder.model_name}")
        print(f"  Max reflections: {self.reflections}")

    # ---------------- Prompt helpers ----------------
    def _build_initial_prompt(self, function_signature: str, content: str, starter_code: str) -> str:
        header = (
            "You will be given a question "
            "and generate a correct Python program that matches the requirement\n\n"
        )
        prompt = header
        prompt += f"### Question:\n{content}\n\n"
        if starter_code:
            prompt += (
                "### Format: You will use the following starter code to write the solution. "
                "Output ONLY a single Python code block. Do not include explanations or tests.\n"
            )
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += (
                "### Format: Implement a function that satisfies the problem. "
            )
            scaffold = f"```python\n{function_signature if function_signature else '# YOUR CODE HERE'}\n```\n\n"
            prompt += scaffold
        prompt += "### Answer: (return ONLY the code inside one Python code block)\n"
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
        prompt = (
            "You are revising a Python solution. "
            "Return the corrected code in one Python code block\n\n"
        )
        prompt += f"### Original Problem\nFunction Signature: {function_signature}\n{content}\n\n"
        if starter_code:
            prompt += f"### Starter Code\n```python\n{starter_code}\n```\n\n"
        prompt += f"### Current Submission (iteration {iteration})\n```python\n{current_code}\n```\n\n"
        prompt += "### Feedback\n" + feedback.strip() + "\n\n"
        prompt += "### Answer: (one Python code block)\n"
        return prompt

    def _build_feedback_prompt(
        self,
        function_signature: str,
        content: str,
        starter_code: str,
        candidate_code: str,
        iteration: int,
    ) -> str:
        prompt = (
            "Review a Python submission and provide feedback.\n"
        )
        prompt += f"### Problem : {function_signature}\n{content}\n\n"
        if starter_code:
            prompt += f"### Starter Code Provided\n```python\n{starter_code}\n```\n\n"
        prompt += f"### Candidate Submission (iteration {iteration})\n```python\n{candidate_code}\n```\n\n"
        prompt += (
            "### Response Format\n"
            "- Verdict: PASS or FAIL with one line justification.\n"
            "- Revision Guidance: for fixing the issues.\n"
            "If the code is correct, still include the PASS verdict and highlight key correctness arguments.\n"
        )
        return prompt

    # ---------------- Utility helpers ----------------
    def _clean_generated_code(self, code: str) -> str:
        code = code.strip()
        if "```python" in code:
            start_idx = code.find("```python") + len("```python")
            while start_idx < len(code) and code[start_idx] in " \t\r\n":
                start_idx += 1
            end_idx = code.find("```", start_idx)
            code = code[start_idx:end_idx if end_idx != -1 else None].strip()
        elif code.startswith("```"):
            lines = code.splitlines()
            filtered = [line for line in lines[1:] if not line.strip().startswith("```")]
            code = "\n".join(filtered).strip()
        return code

    def _extract_function_signature(self, starter_code: str) -> str:
        if not starter_code:
            return ""
        for line in starter_code.splitlines():
            if line.strip().startswith("def "):
                return line.strip()
        return ""

    def _extract_verdict(self, feedback: str) -> str:
        for line in feedback.splitlines():
            upper = line.upper()
            if "VERDICT" in upper:
                if "PASS" in upper:
                    return "PASS"
                if "FAIL" in upper:
                    return "FAIL"
        return "UNKNOWN"

    # ---------------- Core pipeline ----------------
    def generate_reflexion_solution(self, problem: Dict[str, Any], index: int) -> ReflexionResult:
        task_id = self._get_task_id(problem, index)
        print(f"\n🎯 Reflexion Loop for {task_id}")
        problem_desc = problem.get("question_content", "").strip()
        starter_code = problem.get("starter_code", "")
        function_signature = self._extract_function_signature(starter_code)

        initial_prompt = self._build_initial_prompt(function_signature, problem_desc, starter_code)
        current_code = self._clean_generated_code(
            self.coder.generate(
                prompt=initial_prompt,
                max_tokens=4096,
                temperature=self.coder.temperature,
                top_p=self.coder.top_p,
                top_k=self.coder.top_k,
                repetition_penalty=self.coder.repetition_penalty,
            )
        )
        print(f"✅ Initial draft length: {len(current_code)} chars")

        steps: List[ReflexionStep] = []
        combined_feedback = []

        for iteration in range(1, self.reflections + 1):
            feedback_prompt = self._build_feedback_prompt(
                function_signature,
                problem_desc,
                starter_code,
                current_code,
                iteration,
            )
            feedback = self.critic.generate(
                prompt=feedback_prompt,
                max_tokens=2048,
                temperature=0,

            )
            verdict = self._extract_verdict(feedback)
            steps.append(ReflexionStep(iteration=iteration, verdict=verdict, feedback=feedback, code=current_code))
            combined_feedback.append(f"Iteration {iteration} Feedback:\n{feedback.strip()}")
            print(f"📝 Feedback iteration {iteration}: verdict={verdict}")

            if verdict == "PASS":
                print("✅ Critic marked solution as PASS. Stopping early.")
                break

            revision_prompt = self._build_revision_prompt(
                function_signature,
                problem_desc,
                starter_code,
                current_code,
                "\n\n".join(combined_feedback),
                iteration,
            )
            revised_code = self._clean_generated_code(
                self.coder.generate(
                    prompt=revision_prompt,
                    max_tokens=4096,
                    temperature=self.coder.temperature,
                    top_p=self.coder.top_p,
                    top_k=self.coder.top_k,
                    repetition_penalty=self.coder.repetition_penalty,
                )
            )

            if revised_code:
                current_code = revised_code
                print(f"🔁 Iteration {iteration} revision length: {len(current_code)} chars")
            else:
                print("⚠️ Revision step returned empty code; keeping previous submission.")
                break

        reasoning_chain = " → ".join(
            [
                f"Iter {step.iteration}: {step.verdict}"
                for step in steps
            ]
        )
        if not reasoning_chain:
            reasoning_chain = "Initial draft only"

        return ReflexionResult(
            task_id=task_id,
            problem_description=problem_desc,
            starter_code=starter_code,
            final_code=current_code,
            reasoning_chain=reasoning_chain,
            steps=steps,
        )

    # ---------------- Dataset & IO ----------------
    def load_livecodebench_dataset(self, version_tag: str = "v6") -> List[Dict[str, Any]]:
        print(f"📚 Loading LiveCodeBench-Lite dataset ({version_tag})...")
        try:
            dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version_tag)
            problems = list(dataset["test"])
            print(f"Loaded {len(problems)} problems")
            return problems
        except Exception as exc:
            print(f"⚠️ Failed to load dataset ({exc}); falling back to dummy sample.")
            return [
                {
                    "question_id": "dummy_1",
                    "question_content": "Write a function hello_world() that returns 'Hello World'.",
                    "starter_code": "def hello_world():\n    pass",
                }
            ]

    def generate_batch(
        self,
        version_tag: str = "v6",
        max_problems: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> List[ReflexionResult]:
        dataset = self.load_livecodebench_dataset(version_tag)
        if end_index is None:
            end_index = len(dataset)
        end_index = min(end_index, len(dataset))
        if start_index < 0 or start_index >= len(dataset):
            raise ValueError(f"start_index ({start_index}) out of range [0, {len(dataset)})")
        if end_index <= start_index:
            raise ValueError("end_index must be greater than start_index")

        subset = dataset[start_index:end_index]
        if max_problems is not None:
            subset = subset[:max_problems]
        print(f"🚀 Processing {len(subset)} problems (indices {start_index}..{start_index + len(subset) - 1})")

        results: List[ReflexionResult] = []
        for offset, problem in enumerate(subset):
            absolute_index = start_index + offset
            try:
                result = self.generate_reflexion_solution(problem, absolute_index)
                results.append(result)
                print(f"✅ {result.task_id}: final code length {len(result.final_code)} chars")
            except Exception as exc:
                print(f"❌ Error on problem {absolute_index}: {exc}")
                results.append(
                    ReflexionResult(
                        task_id=self._get_task_id(problem, absolute_index),
                        problem_description=problem.get("question_content", ""),
                        starter_code=problem.get("starter_code", ""),
                        final_code=f"# Error: {exc}",
                        reasoning_chain=str(exc),
                        steps=[],
                    )
                )

        timestamp = int(time.time())
        results_path = self.output_dir / f"reflexion_feedback_results_{timestamp}.json"
        livebench_path = self.output_dir / f"reflexion_feedback_livebench_{timestamp}.json"
        self._write_results(results, results_path, livebench_path)
        print(f"\n📦 Saved detailed results to {results_path}")
        print(f"📦 Saved LiveCodeBench format to {livebench_path}")
        return results

    def _write_results(
        self,
        results: List[ReflexionResult],
        results_path: Path,
        livebench_path: Path,
    ) -> None:
        payload = {
            "generation_info": {
                "approach": "Reflexion-Feedback",
                "reflections": self.reflections,
                "critic_model": self.critic.model_name,
                "coder_model": self.coder.model_name,
                "timestamp": int(time.time()),
            },
            "results": [
                {
                    "task_id": r.task_id,
                    "problem_description": r.problem_description,
                    "starter_code": r.starter_code,
                    "final_code": r.final_code,
                    "reasoning_chain": r.reasoning_chain,
                    "steps": [
                        {
                            "iteration": step.iteration,
                            "verdict": step.verdict,
                            "feedback": step.feedback,
                            "code": step.code,
                        }
                        for step in r.steps
                    ],
                }
                for r in results
            ],
        }
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        livebench_entries = []
        for result in results:
            if not result.final_code.startswith("# Error"):
                livebench_entries.append(
                    {"question_id": result.task_id, "code_list": [result.final_code]}
                )
        with open(livebench_path, "w", encoding="utf-8") as handle:
            json.dump(livebench_entries, handle, indent=2)

    # ---------------- Helper methods ----------------
    def _get_task_id(self, problem: Dict[str, Any], index: int) -> str:
        for field in ["question_id", "id", "task_id", "problem_id"]:
            if field in problem:
                return str(problem[field])
        return f"task_{index}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reflexion-style feedback pipeline on LiveCodeBench-Lite")
    parser.add_argument("--critic-model", type=str, default="Qwen/Qwen3-8B", help="Model used for feedback generation")
    parser.add_argument("--critic-base-url", type=str, default="http://localhost:5678/v1", help="Base URL for critic OpenAI-compatible endpoint")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", help="Coder model served by OpenAI-compatible endpoint")
    parser.add_argument("--qwen-base-url", type=str, default=None, help="Base URL for coder OpenAI-compatible endpoint")
    parser.add_argument("--output-dir", type=str, default="reflexion_feedback_results", help="Directory to store outputs")
    parser.add_argument("--version", type=str, default="v6", help="LiveCodeBench dataset version")
    parser.add_argument("--start-index", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=175, help="End index (exclusive)")
    parser.add_argument("--max-problems", type=int, default=None, help="Optional cap on number of problems")
    parser.add_argument("--reflections", type=int, default=1, help="Maximum feedback iterations")
    parser.add_argument("--temperature", type=float, default=0.6, help="Coder sampling temperature")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95, help="Coder nucleus sampling top-p")
    parser.add_argument("--top-k", dest="top_k", type=int, default=20, help="Coder top-k sampling")
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.05, help="Coder repetition penalty")
    parser.add_argument("--critic-temperature", dest="critic_temperature", type=float, default=0.0, help="Critic sampling temperature")
    parser.add_argument("--test-connections", action="store_true", help="Only test critic/coder connectivity")
    parser.add_argument("--eval", action="store_true", help="Run LiveCodeBench evaluation after generation")
    parser.add_argument("--all", action="store_true", help="Ignore --max-problems and process entire slice")
    args = parser.parse_args()

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
        if critic_ok and coder_ok:
            print("✅ Both connections successful.")
        else:
            if not critic_ok:
                print("❌ Critic connection failed.")
            if not coder_ok:
                print("❌ Coder connection failed.")
        return

    generator = ReflexionApproachGenerator(
        critic_client=critic_client,
        coder_client=coder_client,
        output_dir=args.output_dir,
        reflections=args.reflections,
    )

    max_problems = None if args.all else args.max_problems
    results = generator.generate_batch(
        version_tag=args.version,
        max_problems=max_problems,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    print("\n" + "=" * 70)
    print("REFLEXION FEEDBACK SUMMARY")
    print("=" * 70)
    successes = sum(1 for r in results if not r.final_code.startswith("# Error"))
    print(f"Problems processed: {len(results)}")
    print(f"Successful generations: {successes}")
    if results:
        print(f"Success rate: {successes / len(results) * 100:.1f}%")

    if args.eval:
        print("\n🧪 Running LiveCodeBench evaluation...")
        try:
            import glob

            livebench_files = glob.glob(str(Path(args.output_dir) / "reflexion_feedback_livebench_*.json"))
            if livebench_files:
                latest = max(livebench_files)
                print(f"Evaluating {latest}")
                generator._run_livecodebench_evaluation(latest, args.version)
            else:
                print("⚠️ No LiveCodeBench outputs available for evaluation.")
        except Exception as exc:
            print(f"❌ Evaluation failed: {exc}")

    print("\nDone.")


def _run_livecodebench_evaluation_placeholder(self, *args, **kwargs):
    print("LiveCodeBench evaluation helper not implemented in this script.")
    print("Use lean_gen/fdg_approach_qwen3_inter_refined.py for a reference implementation if needed.")


# Attach placeholder as instance method to mirror ITSSM script capability without duplicating logic.
setattr(ReflexionApproachGenerator, "_run_livecodebench_evaluation", _run_livecodebench_evaluation_placeholder)


if __name__ == "__main__":
    main()
