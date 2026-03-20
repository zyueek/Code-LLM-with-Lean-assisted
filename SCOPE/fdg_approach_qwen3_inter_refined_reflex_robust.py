#!/usr/bin/env python3
"""
ITSSM (Initial Term Synthesis, Subgoal Mapping) — Qwen3 + Robust Reflexion

Goal: Improve overall LiveCodeBench generation quality without relying on past
evaluation files. We keep the original coder prompt structure and avoid
multi-sampling, but add a robust feedback loop:

- Stage 1: Coder drafts code (strict Qwen prompt, single code block)
- Stage 2a: Prover type analysis using ONLY problem description + code
- Stage 2b: Prover robustness checklist (format strictness, edge-cases,
            complexity/algorithmic hints) using ONLY problem description + code
- Stage 3: Coder revision (same revision prompt structure)
- Stage 4: Optional reflexion iteration (same structure), feedback from prover
           restricted to problem description + code

Coder defaults are tuned for diversity (temperature=0.6, top_p=0.95).
Static cleanup removes any accidental __main__ blocks.
"""

import json
import os
import sys
import subprocess
import time
import argparse
import re
import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "EMPTY")

# Add parent directory to path to import vllm_client
sys.path.insert(0, str(Path(__file__).parent.parent))
from vllm_client import VLLMClient

try:
    import openai
except ImportError:
    print("Installing openai library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
    import openai

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
    from datasets import load_dataset


class OpenAIClient:
    """Client for vLLM Qwen3 API (OpenAI-compatible)"""

    def __init__(
        self,
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("QWEN_VLLM_BASE_URL", "http://localhost:1234/v1"),
        )
        self.model_name = model_name
        # Default decoding params for Qwen via vLLM
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        print(f"Initialized vLLM Qwen3 client with model {self.model_name}")

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
        """Generate text completion using vLLM Qwen3"""
        try:
            print(f"🔍 Sending prompt to vLLM Qwen3 (first 160 chars): {prompt[:160]}...")
            # Resolve params (call overrides -> defaults)
            t = self.temperature if temperature is None else temperature
            tp = self.top_p if top_p is None else top_p
            tk = self.top_k if top_k is None else top_k
            rp = self.repetition_penalty if repetition_penalty is None else repetition_penalty

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # vLLM OpenAI-compatible endpoints generally expect 'max_tokens'
                max_tokens=max_tokens,
                temperature=t,
                top_p=tp,
                stop=stop,
                # Non-standard OpenAI params (vendor-specific) go via extra_body
                extra_body={
                    "top_k": tk,
                    "repetition_penalty": rp,
                },
            )

            generated_text = response.choices[0].message.content.strip()
            print(f"✅ vLLM Qwen3 generated {len(generated_text)} characters")
            return generated_text

        except Exception as e:
            print(f"❌ Error generating with vLLM Qwen3: {e}")
            return ""

    def test_connection(self) -> bool:
        try:
            test_response = self.generate("def hello():", max_tokens=10, temperature=0)
            if test_response:
                print(f"✅ vLLM Qwen3 connection test successful. Response: {test_response[:50]}...")
                return True
            else:
                print("⚠️ vLLM Qwen3 API accessible but generation failed")
                return False
        except Exception as e:
            print(f"⚠️ vLLM Qwen3 API not accessible: {e}")
            return False


@dataclass
class ITSSMResult:
    task_id: str
    problem_description: str
    initial_proof_term: str
    type_analysis: str
    revised_python: str
    full_function: str
    reasoning_chain: str


class ITSSMApproachGenerator:
    """ITSSM with robust reflexion (no multi-sample, original coder prompts)"""

    def __init__(
        self,
        prover_client: VLLMClient,
        coder_client,
        output_dir: str = "./itssm_approach_results",
        reflections: int = 1,
        coder_max_tokens: int = 15000,
    ):
        self.prover = prover_client
        self.coder = coder_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reflections = max(0, int(reflections))
        self.coder_max_tokens = max(1, int(coder_max_tokens))

        print("🧠 ITSSM Robust Reflex initialized:")
        print(f"  Prover LLM: {self.prover.vllm_url} ({self.prover.model_name})")
        if hasattr(self.coder, 'model_name'):
            print(f"  Coder LLM: {self.coder.model_name}")
        else:
            print(f"  Coder LLM: {type(self.coder).__name__}")
        print(f"  Reflexion iterations: {self.reflections}")
        print(f"  Coder max_tokens: {self.coder_max_tokens}")

    # ---------------- Qwen-style prompt builders (unchanged structure) ----------------
    def _build_qwen_initial_prompt(self, function_signature: str, content: str, starter_code: str) -> str:
        header = (
            "You are an expert Python programmer. You will be given a question (problem specification) "
            "and must generate a correct Python program that matches the specification and passes all tests.\n\n"
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
                "Output ONLY a single Python code block. Do not include explanations or tests.\n"
            )
            scaffold = f"```python\n{function_signature if function_signature else '# YOUR CODE HERE'}\n```\n\n"
            prompt += scaffold
        prompt += "### Answer: (return ONLY the code inside one Python code block)\n"
        return prompt

    def _build_qwen_revision_prompt(self, function_signature: str, content: str, initial_code: str, type_analysis: str, starter_code: str) -> str:
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
            "You may introduce small helpers if necessary, but keep the solution minimal and clear.\n\n"
        )
        prompt += "### Answer: (one Python code block, no extra text)\n"
        return prompt

    # ---------------- Prover helpers (restricted input) ----------------
    def _prover_type_analysis(self, problem: Dict[str, Any], code: str) -> str:
        content = problem.get("question_content", "")
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
        return self.prover.generate(prompt, max_tokens=4096, temperature=0.0)

    def _prover_robustness_checklist(self, problem: Dict[str, Any], code: str) -> str:
        content = problem.get("question_content", "")
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
        return self.prover.generate(prompt, max_tokens=512, temperature=0.0)

    def _prover_feedback(self, problem: Dict[str, Any], code: str) -> str:
        content = problem.get("question_content", "")
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
        return self.prover.generate(prompt, max_tokens=600, temperature=0.0)

    # ---------------- Utilities ----------------
    def _clean_generated_code(self, code: str) -> str:
        code = code.strip()
        if '```python' in code:
            i = code.find('```python') + len('```python')
            while i < len(code) and code[i] in ' \t\r\n':
                i += 1
            j = code.find('```', i)
            code = code[i:j].strip() if j != -1 else code[i:].strip()
        elif '```' in code:
            lines = code.split('\n')
            buff, in_block = [], False
            for line in lines:
                if line.strip().startswith('```'):
                    in_block = not in_block
                elif in_block:
                    buff.append(line)
            if buff:
                code = '\n'.join(buff).strip()
        if code.startswith('```'):
            code = '\n'.join(code.split('\n')[1:]).strip()
        if code.endswith('```'):
            code = '\n'.join(code.split('\n')[:-1]).strip()
        return code

    def _strip_main_block(self, code: str) -> str:
        # Remove if __name__ == '__main__': blocks conservatively
        pattern = re.compile(r"if __name__ == ['\"]__main__['\"]:\n(?:    .*(?:\n|$))+", re.MULTILINE)
        return re.sub(pattern, '', code)

    # ---------------- Stages ----------------
    def stage1_initial(self, problem: Dict[str, Any]) -> str:
        content = problem.get("question_content", "")
        starter_code = problem.get("starter_code", "")
        function_signature = ""
        if starter_code:
            for line in starter_code.split('\n'):
                t = line.strip()
                if t.startswith('def ') or t.startswith('class '):
                    function_signature = t
                    break
        prompt = self._build_qwen_initial_prompt(function_signature, content, starter_code)
        draft = self.coder.generate(prompt, max_tokens=self.coder_max_tokens)
        draft = self._clean_generated_code(draft)
        print(f"✅ Draft length: {len(draft)} chars")
        return draft

    def stage2_analysis_and_checklist(self, problem: Dict[str, Any], code: str) -> str:
        analysis = self._prover_type_analysis(problem, code)
        checklist = self._prover_robustness_checklist(problem, code)
        combined = analysis.strip() + "\n\nRobustness Checklist:\n" + checklist.strip()
        return combined

    def stage3_revision(self, problem: Dict[str, Any], current_code: str, combined_analysis: str) -> str:
        content = problem.get("question_content", "")
        starter_code = problem.get("starter_code", "")
        function_signature = ""
        if starter_code:
            for line in starter_code.split('\n'):
                t = line.strip()
                if t.startswith('def ') or t.startswith('class '):
                    function_signature = t
                    break
        prompt = self._build_qwen_revision_prompt(
            function_signature=function_signature,
            content=content,
            initial_code=current_code,
            type_analysis=combined_analysis,
            starter_code=starter_code,
        )
        revised = self.coder.generate(prompt, max_tokens=self.coder_max_tokens)
        revised = self._clean_generated_code(revised)
        revised = self._strip_main_block(revised)
        print(f"✅ Revised length: {len(revised)} chars")
        return revised

    def generate_one(self, problem: Dict[str, Any], index: int) -> ITSSMResult:
        task_id = self.get_task_id(problem, index)
        print(f"\n🎯 ITSSM Robust Reflex for {task_id}")

        draft = self.stage1_initial(problem)
        combined = self.stage2_analysis_and_checklist(problem, draft)
        revised = self.stage3_revision(problem, draft, combined)

        for r in range(self.reflections):
            print(f"♻️ Reflexion {r+1}/{self.reflections}")
            fb = self._prover_feedback(problem, revised)
            combined2 = combined + "\n\nReflexion Feedback:\n" + fb.strip()
            new_revised = self.stage3_revision(problem, revised, combined2)
            if new_revised.strip() == revised.strip():
                print("🔁 No effective change; stop.")
                break
            revised = new_revised
            combined = combined2

        result = ITSSMResult(
            task_id=task_id,
            problem_description=problem.get("question_content", ""),
            initial_proof_term=draft,
            type_analysis=combined,
            revised_python=revised,
            full_function=revised,
            reasoning_chain=f"draft→analysis+checklist→revise→reflex x{self.reflections}",
        )
        return result

    # ---------------- Dataset / batch / utils ----------------
    def get_task_id(self, problem: Dict[str, Any], index: int) -> str:
        for field in ['question_id', 'id', 'task_id', 'problem_id']:
            if field in problem:
                return str(problem[field])
        return f"task_{index}"

    def load_livecodebench_dataset(self, version_tag: str = "v6") -> List[Dict[str, Any]]:
        print(f"Loading LiveCodeBench-Lite dataset ({version_tag})...")
        try:
            dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version_tag)
            dataset_list = list(dataset['test'])
            print(f"Loaded {len(dataset_list)} problems")
            return dataset_list
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            print("Creating dummy dataset for testing...")
            return [
                {
                    "question_id": "test_1",
                    "question_title": "Test Problem 1",
                    "question_content": "Write a function that returns 'Hello World'",
                    "starter_code": "def hello_world():\n    pass",
                }
            ]

    def generate_batch(
        self,
        max_problems: Optional[int] = None,
        version_tag: str = "v6",
        start_index: int = 0,
        end_index: Optional[int] = None,
        run_timestamp: Optional[int] = None,
    ) -> List[ITSSMResult]:
        print("🚀 Starting ITSSM Robust Reflex generation...")
        dataset = self.load_livecodebench_dataset(version_tag)

        if end_index is None:
            end_index = len(dataset)
        if start_index < 0 or start_index >= len(dataset):
            raise ValueError(f"start_index ({start_index}) out of range [0, {len(dataset)})")
        if end_index < start_index or end_index > len(dataset):
            raise ValueError(f"end_index ({end_index}) must be >= start_index ({start_index}) and <= {len(dataset)}")

        dataset = dataset[start_index:end_index]
        print(f"Processing problems {start_index} to {end_index-1} ({len(dataset)} problems)")
        if max_problems and max_problems < len(dataset):
            dataset = dataset[:max_problems]
            print(f"Limited to {len(dataset)} problems due to max_problems")

        results: List[ITSSMResult] = []
        start_time = time.time()
        for i, problem in enumerate(dataset):
            original_index = start_index + i
            task_id = self.get_task_id(problem, original_index)
            print(f"\n--- Problem {original_index+1} ({i+1}/{len(dataset)}): {task_id} ---")
            try:
                res = self.generate_one(problem, original_index)
                results.append(res)
                print(f"✅ {task_id}: Revised length {len(res.revised_python)} chars")
            except Exception as e:
                print(f"❌ Error processing {task_id}: {e}")
                results.append(ITSSMResult(
                    task_id=task_id,
                    problem_description=problem.get("question_content", ""),
                    initial_proof_term=f"# Error: {e}",
                    type_analysis=f"# Error: {e}",
                    revised_python=f"# Error: {e}",
                    full_function=f"# Error: {e}",
                    reasoning_chain=f"Error: {e}",
                ))

        end_time = time.time()

        timestamp = int(run_timestamp) if run_timestamp is not None else int(time.time())
        if timestamp <= 0:
            raise ValueError(f"run_timestamp must be positive when provided; got {timestamp}")
        results_file = self.output_dir / f"itssm_robust_reflex_results_{timestamp}.json"
        results_data = {
            "generation_info": {
                "approach": "ITSSM_robust_reflex",
                "framework": "Initial Term Synthesis + Robust Reflexion",
                "num_problems": len(results),
                "generation_time_seconds": end_time - start_time,
                "prover_model": self.prover.model_name,
                "coder_model": getattr(self.coder, 'model_name', 'unknown'),
                "reflections": self.reflections,
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
                } for r in results
            ],
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        livebench_file = self.output_dir / f"itssm_robust_reflex_livebench_{timestamp}.json"
        livebench_data = []
        for r in results:
            if not r.revised_python.startswith("# Error"):
                livebench_data.append({
                    "question_id": r.task_id,
                    "code_list": [r.revised_python],
                })
        with open(livebench_file, 'w') as f:
            json.dump(livebench_data, f, indent=2)

        print(f"\n🎉 ITSSM Robust Reflex generation complete!")
        print(f"Detailed results: {results_file}")
        print(f"LiveCodeBench format: {livebench_file}")
        return results

    # ---------------- Evaluation (optional) ----------------
    def run_livecodebench_evaluation(self, output_file: str, release_version: str):
        print(f"\n--- Running LiveCodeBench Evaluation ---")
        try:
            import lcb_runner
            print("LiveCodeBench framework found, running evaluation...")
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="lcb_eval_")
            cmd = [
                sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
                "--custom_output_file", output_file,
                "--output_dir", temp_dir,
                "--release_version", release_version,
            ]
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("LiveCodeBench evaluation completed successfully!")
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            results_file = os.path.join(temp_dir, "eval_all.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"Evaluation results: {results}")
                return results, temp_dir
            else:
                print("Results file not found, but evaluation completed")
                return None, temp_dir
        except ImportError:
            print("LiveCodeBench framework not found.")
            print("Please install it with: pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git")
            return None, None
        except Exception as e:
            print(f"LiveCodeBench evaluation failed: {e}")
            return None, None


def main():
    parser = argparse.ArgumentParser(description="ITSSM (Qwen-style prompts) with Robust Reflexion")
    parser.add_argument("--prover-url", type=str, default=os.getenv("PROVER_BASE_URL", "http://localhost:5678/"), help="Prover (DeepSeek Prover) OpenAI-compatible base URL")
    parser.add_argument("--output-dir", type=str, default="./itssm_approach_results", help="Directory to store outputs")
    parser.add_argument("--version", type=str, default="v6", help="LiveCodeBench dataset version tag")
    parser.add_argument("--eval", action="store_true", help="Run LiveCodeBench evaluation after generation")
    parser.add_argument("--test-connections", action="store_true", help="Only test model endpoints and exit")
    parser.add_argument("--all", action="store_true", help="Ignore --max-problems and process entire range")
    parser.add_argument("--max-problems", type=int, default=None, help="Max problems to process")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in dataset (inclusive)")
    parser.add_argument("--end-index", type=int, default=175, help="End index in dataset (exclusive)")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", help="Qwen model name for vLLM endpoint")
    parser.add_argument("--qwen-base-url", type=str, default=None, help="Override vLLM base URL (default http://localhost:1234/v1)")
    parser.add_argument(
        "--coder-max-tokens",
        type=int,
        default=15000,
        help="Max tokens for coder generation in stage-1/stage-3.",
    )
    # Decoding params
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", dest="top_k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.05, help="Repetition penalty (>=1.0)")
    # Reflexion knob
    parser.add_argument("--reflections", type=int, default=1, help="Number of reflexion iterations (>=0)")
    parser.add_argument(
        "--run-timestamp",
        type=int,
        default=None,
        help="Optional fixed UNIX timestamp used for output filenames (reproducible naming).",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("\n🎯 ITSSM with Qwen-style Prompts + Robust Reflexion")
        print("=" * 60)
        print("Adds robustness checklist; keeps coder prompts unchanged; no multi-sample")
        print()
        print("Examples:")
        print("  python fdg_approach_qwen3_inter_refined_reflex_robust.py --test-connections")
        print("  python fdg_approach_qwen3_inter_refined_reflex_robust.py --max-problems 10 --version v6 --reflections 1")
        print()
        parser.print_help()
        return

    if args.qwen_base_url:
        os.environ["QWEN_VLLM_BASE_URL"] = args.qwen_base_url

    print("🔗 Connecting to servers...")
    prover_client = VLLMClient(base_url=args.prover_url, model_name="deepseek-ai/DeepSeek-Prover-V2-7B")
    coder_client = OpenAIClient(
        api_key="EMPTY",
        model_name=args.qwen_model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
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
                print(f"  - Prover server ({args.prover_url}) not accessible")
            if not coder_ok:
                print(f"  - Qwen vLLM API not accessible")
        return

    generator = ITSSMApproachGenerator(
        prover_client,
        coder_client,
        args.output_dir,
        reflections=args.reflections,
        coder_max_tokens=args.coder_max_tokens,
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
        print(f"\n🧪 Running LiveCodeBench evaluation...")
        import glob
        livebench_files = glob.glob(str(generator.output_dir / "itssm_robust_reflex_livebench_*.json"))
        if livebench_files:
            latest_file = max(livebench_files)
            print(f"Evaluating: {latest_file}")
            try:
                eval_results, temp_dir = generator.run_livecodebench_evaluation(latest_file, args.version)
                if eval_results:
                    results_file = latest_file.replace('.json', '_results.json')
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(eval_results, f, indent=2)
                    print(f"Evaluation results saved to: {results_file}")
                    print("✅ Evaluation completed!")
                else:
                    print("⚠️ Evaluation completed but no results file generated")
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                print("💡 Make sure LiveCodeBench is installed: pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git")
        else:
            print("❌ No LiveCodeBench files found to evaluate")


if __name__ == "__main__":
    main()
