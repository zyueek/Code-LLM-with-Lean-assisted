#!/usr/bin/env python3
"""Simple vLLM Qwen3 BigCodeBench Code Generation.

This script mirrors the behaviour of ``simple_qwen3_livebench.py`` but switches
the evaluation benchmark from LiveCodeBench to **BigCodeBench**. It keeps the
same Qwen3 model settings and simple prompting style while loading
BigCodeBench tasks and writing outputs directly in the BigCodeBench
``samples.jsonl`` format so they can be evaluated with ``bigcodebench.evaluate``.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).parent

# Import the original simple Qwen3 generator (same settings, same prompt).
import simple_qwen3_livebench as qwen_live

# Ensure local BigCodeBench repo is importable.
BIGCODEBENCH_ROOT = BASE_DIR / "bigcodebench"
if BIGCODEBENCH_ROOT.exists():
    sys.path.insert(0, str(BIGCODEBENCH_ROOT))

try:
    from bigcodebench.data import get_bigcodebench
    from bigcodebench.sanitize import sanitize
except Exception as exc:  # pragma: no cover - depends on local BigCodeBench setup
    raise RuntimeError(
        "BigCodeBench repo is not importable. Make sure it is cloned at "
        "'lean_gen/bigcodebench' and installed with:\n"
        "  cd lean_gen/bigcodebench && pip install -e .\n"
    ) from exc


def load_bigcodebench_tasks(split: str = "instruct", subset: str = "hard") -> List[Dict[str, Any]]:
    """Load BigCodeBench tasks and expose them for Qwen3 generation.

    - ``instruct`` split uses ``instruct_prompt`` as the natural-language question.
    - ``complete`` split uses ``complete_prompt``.
    """

    print(f"📊 Loading BigCodeBench dataset (split={split}, subset={subset})...")
    dataset = get_bigcodebench(subset=subset)
    tasks: List[Dict[str, Any]] = []

    for task_id, task in dataset.items():
        if split == "instruct":
            question = task.get("instruct_prompt", "")
        elif split == "complete":
            question = task.get("complete_prompt", "")
        else:
            raise ValueError("split must be 'instruct' or 'complete'")

        tasks.append(
            {
                "task_id": task_id,
                "question": question,
                "entry_point": task.get("entry_point", ""),
            }
        )

    print(f"✅ Loaded {len(tasks)} BigCodeBench tasks")
    return tasks


def generate_for_bigcodebench(
    generator: qwen_live.SimpleGPT4Generator,
    tasks: List[Dict[str, Any]],
    start_index: int = 0,
    end_index: Optional[int] = None,
    output_path: Path = Path("bcb_results/simple_qwen3_bigcodebench_samples.jsonl"),
) -> bool:
    """Run simple Qwen3 generation on BigCodeBench and write samples.jsonl.

    Output records follow BigCodeBench's expected schema:
        {"task_id": ..., "solution": ..., "raw_solution": ...}
    """

    if end_index is None:
        end_index = len(tasks)

    if start_index < 0 or start_index >= len(tasks):
        print(f"❌ start_index ({start_index}) out of range [0, {len(tasks)})")
        return False
    if end_index < start_index or end_index > len(tasks):
        print(f"❌ end_index ({end_index}) must be >= start_index ({start_index}) and <= {len(tasks)}")
        return False

    selected = tasks[start_index:end_index]
    print(
        f"🚀 Generating for {len(selected)} BigCodeBench tasks "
        f"(indices {start_index}..{end_index - 1})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    written = 0

    with output_path.open("w", encoding="utf-8") as out:
        for idx, task in enumerate(selected):
            task_id = task["task_id"]
            question = task["question"]
            entry_point = task.get("entry_point", "")

            print(f"\n--- Task {start_index + idx} ({task_id}) ---")
            try:
                # Reuse the original interface: pass a problem dict with question_content.
                problem = {"question_content": question}
                raw_code = generator.generate_code(problem)

                # Clean markdown fences using the original helper (if present)
                clean_code = raw_code
                if hasattr(generator, "_clean_generated_code") and isinstance(raw_code, str):
                    clean_code = generator._clean_generated_code(raw_code)

                if clean_code:
                    try:
                        solution = sanitize(clean_code, entry_point or None)
                    except Exception:
                        solution = clean_code
                    print(f"✅ Generated {len(clean_code)} characters")
                else:
                    solution = ""
                    print("❌ Empty generation")

                record = {
                    "task_id": task_id,
                    "solution": solution if solution.strip() else "# Error: empty generation",
                    "raw_solution": raw_code,
                }
                out.write(json.dumps(record) + "\n")
                written += 1
            except Exception as exc:
                print(f"❌ Error on task {task_id}: {exc}")
                record = {
                    "task_id": task_id,
                    "solution": f"# Error during generation: {exc}",
                    "raw_solution": "",
                }
                out.write(json.dumps(record) + "\n")
                written += 1

    elapsed = time.time() - start_time
    print("\n🎉 Generation complete!")
    print(f"✅ Samples saved: {output_path} (tasks written: {written})")
    print(f"⏱️ Time elapsed: {elapsed:.1f} seconds")
    print("\n📋 Next steps (local evaluation):")
    print(
        "  bigcodebench.evaluate --execution local "
        f"--split instruct --subset hard --samples {output_path}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple vLLM Qwen3 code generation for BigCodeBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test vLLM Qwen3 connection
  python simple_qwen3_bigcodebench.py --test-connection

  # Generate for first 50 BigCodeBench-Hard instruct tasks
  python simple_qwen3_bigcodebench.py --bcb-split instruct --bcb-subset hard \
    --start-index 0 --end-index 50 --output bcb_results/simple_qwen3_hard_instruct.jsonl
        """,
    )

    # API / model options (same defaults as simple_qwen3_livebench)
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var, optional for vLLM)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        help="Model name (default: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)",
    )

    # BigCodeBench options
    parser.add_argument(
        "--bcb-split",
        type=str,
        default="instruct",
        choices=["instruct", "complete"],
        help="BigCodeBench split: instruct or complete",
    )
    parser.add_argument(
        "--bcb-subset",
        type=str,
        default="hard",
        choices=["full", "hard"],
        help="BigCodeBench subset: full or hard",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in BigCodeBench tasks (default: 0)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="End index in BigCodeBench tasks (exclusive)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="bcb_results/simple_qwen3_bigcodebench_samples.jsonl",
        help="Output samples JSONL file (BigCodeBench format)",
    )

    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test vLLM Qwen3 connection and exit",
    )

    args = parser.parse_args()

    # Qwen3 script treats API key as optional and uses vLLM via base_url.
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

    print("🎯 Simple vLLM Qwen3 BigCodeBench Code Generation")
    print("=" * 60)

    generator = qwen_live.SimpleGPT4Generator(api_key, args.model)

    if args.test_connection:
        if generator.test_connection():
            print("✅ Ready for code generation!")
        else:
            print("❌ API connection failed!")
        return

    tasks = load_bigcodebench_tasks(split=args.bcb_split, subset=args.bcb_subset)
    success = generate_for_bigcodebench(
        generator=generator,
        tasks=tasks,
        start_index=args.start_index,
        end_index=args.end_index,
        output_path=Path(args.output),
    )

    if not success:
        print("❌ Generation failed")


if __name__ == "__main__":
    main()
