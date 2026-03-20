#!/usr/bin/env python3
"""
Simple GPT-4.1 Mini LiveCodeBench Code Generation

This script generates code for LiveCodeBench v6 problems using qwen3 with simple prompts.
No special approaches - just direct problem-to-code generation.

Usage:
    python simple_gpt4_livebench.py --start-index 0 --end-index 10 --output results.json
    python simple_gpt4_livebench.py --v6-only --output v6_results.json
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
from datasets import load_dataset

class SimpleGPT4Generator:
    """Simple code generator using vLLM Qwen3."""

    def __init__(self, api_key: str = "EMPTY", model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"):
        """Initialize the vLLM Qwen3 client."""
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="http://localhost:1234/v1"
        )
        self.model_name = model_name
        print(f"🤖 Initialized vLLM Qwen3 client: {self.model_name}")
    
    def create_simple_prompt(self, problem: Dict[str, Any]) -> str:
        """Create a simple, direct prompt for code generation."""
        question_content = problem.get("question_content", "")
        
        # Simple prompt template
        prompt = f"""You are a Python programming expert. Please solve this coding problem.

Problem:
{question_content}

Make sure the code can directly use the sample input and print follow the sample output, calls the function to remember to execute the code.
Return only the Python code without any explanation or markdown formatting."""
        
        return prompt
    
    def generate_code(self, problem: Dict[str, Any]) -> str:
        """Generate code for a single problem."""
        prompt = self.create_simple_prompt(problem)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2048,
                temperature=0 # Use default temperature for GPT-4.1 mini
            )
            
            generated_code = response.choices[0].message.content.strip()
            return generated_code
            
        except Exception as e:
            print(f"❌ Error generating code: {e}")
            return ""
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code"""
        code = code.strip()
        
        # Remove markdown code blocks if present
        # Handle ```python blocks first
        if '```python' in code:
            start_idx = code.find('```python')
            # Find the content after ```python
            start_content = start_idx + len('```python')
            # Skip any whitespace/newlines after ```python
            while start_content < len(code) and code[start_content] in ' \t\r\n':
                start_content += 1
            
            # Find the closing ```
            end_idx = code.find('```', start_content)
            if end_idx != -1:
                code = code[start_content:end_idx].strip()
            else:
                # No closing ```, take everything after ```python
                code = code[start_content:].strip()
        
        # Handle other ``` blocks
        elif '```' in code:
            lines = code.split('\n')
            clean_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```'):
                    if not in_code_block:
                        in_code_block = True
                    else:
                        in_code_block = False
                elif in_code_block:
                    clean_lines.append(line)
            
            if clean_lines:
                code = '\n'.join(clean_lines).strip()
        
        # Remove any remaining ``` at start or end
        if code.startswith('```'):
            code = '\n'.join(code.split('\n')[1:]).strip()
        if code.endswith('```'):
            code = '\n'.join(code.split('\n')[:-1]).strip()
        
        return code    
    def test_connection(self) -> bool:
        """Test if the API is working."""
        try:
            test_problem = {
                "question_content": "Write a function that returns the sum of two numbers.",
                "starter_code": "def add_numbers(a, b):\n    pass"
            }
            test_code = self.generate_code(test_problem)
            if test_code:
                print(f"✅ API connection successful")
                return True
            else:
                print("❌ API connection failed - no response")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False


def load_livecodebench_v6_dataset() -> List[Dict]:
    """Load the LiveCodeBench v6 dataset."""
    try:
        print("📊 Loading LiveCodeBench dataset...")
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag="v6", split="test")
        dataset_list = list(dataset)
        print(f"✅ Loaded {len(dataset_list)} problems")
        return dataset_list
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return []


def filter_v6_only_problems(dataset: List[Dict]) -> List[Dict]:
    """Filter to only v6 problems (from 2025-01-01 onwards)."""
    v6_problems = []
    for item in dataset:
        contest_date = item.get('contest_date', '')
        if contest_date >= '2025-01-01':
            v6_problems.append(item)
    
    print(f"📊 Filtered to v6-only problems: {len(v6_problems)} problems (from 2025-01-01 onwards)")
    return v6_problems


def generate_for_dataset(generator: SimpleGPT4Generator, 
                        dataset: List[Dict],
                        start_index: int = 0,
                        end_index: Optional[int] = None,
                        output_file: str = "simple_gpt4_results.json") -> bool:
    """Generate code for a dataset range."""
    
    if end_index is None:
        end_index = len(dataset)
    
    # Validate indices
    if start_index < 0 or start_index >= len(dataset):
        print(f"❌ start_index ({start_index}) out of range [0, {len(dataset)})")
        return False
    if end_index < start_index or end_index > len(dataset):
        print(f"❌ end_index ({end_index}) must be >= start_index ({start_index}) and <= {len(dataset)}")
        return False
    
    # Slice dataset
    problems = dataset[start_index:end_index]
    print(f"🎯 Generating code for {len(problems)} problems (indices {start_index}-{end_index-1})")
    
    results = []
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question_id = problem.get('question_id', f'unknown_{start_index + i}')
        print(f"\n--- Problem {start_index + i + 1} ({i+1}/{len(problems)}): {question_id} ---")
        
        try:
            # Generate code
            generated_code = generator.generate_code(problem)
            print(generated_code)
            generated_code= generator._clean_generated_code(generated_code)
            print("\n--- Cleaned Generated Code ---")
            print(generated_code)
            if generated_code:
                # Create result entry in LiveCodeBench format
                result = {
                    "question_id": question_id,
                    "code_list": [generated_code]
                }
                results.append(result)
                print(f"✅ Generated {len(generated_code)} characters")
            else:
                # Create empty entry for failed generation
                result = {
                    "question_id": question_id,
                    "code_list": [""]
                }
                results.append(result)
                print(f"❌ Generation failed")
                
        except Exception as e:
            print(f"❌ Error processing {question_id}: {e}")
            # Create empty entry for errors
            result = {
                "question_id": question_id,
                "code_list": [""]
            }
            results.append(result)
    
    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        success_count = sum(1 for r in results if r['code_list'][0])
        
        print(f"\n🎉 Generation complete!")
        print(f"✅ Output saved: {output_file}")
        print(f"📊 Results: {success_count}/{len(results)} successful")
        print(f"⏱️ Time elapsed: {elapsed_time:.1f} seconds")
        print(f"🚀 Ready for evaluation with lcb_runner")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False


def create_v6_evaluation_dataset(input_file: str, output_file: str) -> bool:
    """Create a v6-only evaluation dataset with empty placeholders."""
    try:
        print(f"📋 Creating v6 evaluation dataset...")
        
        # Load input results
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        print(f"   Input problems: {len(input_data)}")
        
        # Load full v6 dataset
        full_dataset = load_livecodebench_v6_dataset()
        v6_problems = filter_v6_only_problems(full_dataset)
        
        # Create mapping
        input_map = {entry['question_id']: entry['code_list'] for entry in input_data}
        
        # Create evaluation dataset
        eval_data = []
        for problem in v6_problems:
            question_id = problem.get('question_id', 'unknown')
            
            if question_id in input_map:
                # Use generated code
                eval_data.append({
                    "question_id": question_id,
                    "code_list": input_map[question_id]
                })
            else:
                # Empty placeholder
                eval_data.append({
                    "question_id": question_id,
                    "code_list": [""]
                })
        
        # Save evaluation dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        filled_count = sum(1 for entry in eval_data if entry['code_list'][0])
        empty_count = len(eval_data) - filled_count
        
        print(f"✅ V6 evaluation dataset created: {output_file}")
        print(f"   Total problems: {len(eval_data)}")
        print(f"   With code: {filled_count}")
        print(f"   Empty placeholders: {empty_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating evaluation dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple GPT-4.1 mini code generation for LiveCodeBench v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test API connection
  python simple_gpt4_livebench.py --test-connection

  # Generate for specific range
  python simple_gpt4_livebench.py --start-index 0 --end-index 10 --output results.json

  # Generate only for v6 problems
  python simple_gpt4_livebench.py --v6-only --start-index 0 --end-index 5 --output v6_results.json

  # Create evaluation dataset
  python simple_gpt4_livebench.py --create-eval-dataset results.json --output eval_ready.json
        """
    )
    
    # API options
    parser.add_argument("--openai-api-key", type=str, 
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
                       help="Model name (default: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)")
    
    # Dataset options
    parser.add_argument("--start-index", type=int, default=0,
                       help="Start index in dataset (default: 0)")
    parser.add_argument("--end-index", type=int, default=175,
                       help="End index in dataset (exclusive)")
    parser.add_argument("--v6-only", action="store_true",
                       help="Use only v6 problems (from 2025-01-01 onwards)")
    
    # Output options
    parser.add_argument("--output", "-o", type=str, default="simple_gpt4_results.json",
                       help="Output JSON file")
    
    # Actions
    parser.add_argument("--test-connection", action="store_true",
                       help="Test API connection and exit")
    parser.add_argument("--create-eval-dataset", type=str,
                       help="Create v6 evaluation dataset from input file")
    
    args = parser.parse_args()
    
    # Get API key (not required for vLLM)
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    
    print("🎯 Simple vLLM Qwen3 LiveCodeBench Code Generation")
    print("=" * 60)
    
    # Special action: create evaluation dataset
    if args.create_eval_dataset:
        success = create_v6_evaluation_dataset(args.create_eval_dataset, args.output)
        if success:
            print(f"\n📋 Next steps:")
            print(f"   python -m lcb_runner.runner.custom_evaluator \\")
            print(f"     --custom_output_file {args.output} \\")
            print(f"     --scenario codegeneration \\")
            print(f"     --release_version v6")
        return
    
    # Initialize generator
    generator = SimpleGPT4Generator(api_key, args.model)
    
    # Test connection
    if args.test_connection:
        if generator.test_connection():
            print("✅ Ready for code generation!")
        else:
            print("❌ API connection failed!")
        return
    
    # Load dataset
    dataset = load_livecodebench_v6_dataset()
    if not dataset:
        print("❌ Failed to load dataset")
        return
    
    # Filter to v6-only if requested
    if args.v6_only:
        dataset = filter_v6_only_problems(dataset)
        if not dataset:
            print("❌ No v6 problems found")
            return
    
    # Generate code
    success = generate_for_dataset(
        generator=generator,
        dataset=dataset,
        start_index=args.start_index,
        end_index=args.end_index,
        output_file=args.output
    )
    
    if success:
        print(f"\n📋 Next steps:")
        if args.v6_only:
            print(f"   1. Create evaluation dataset:")
            print(f"      python {sys.argv[0]} --create-eval-dataset {args.output} --output eval_ready.json")
            print(f"   2. Run evaluation:")
            print(f"      python -m lcb_runner.runner.custom_evaluator \\")
            print(f"        --custom_output_file eval_ready.json \\")
            print(f"        --scenario codegeneration \\")
            print(f"        --release_version v6")
        else:
            print(f"   Run evaluation:")
            print(f"     python -m lcb_runner.runner.custom_evaluator \\")
            print(f"       --custom_output_file {args.output} \\")
            print(f"       --scenario codegeneration")


if __name__ == "__main__":
    main()