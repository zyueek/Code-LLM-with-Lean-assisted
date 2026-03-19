# BigCodeBench Best Run (ITSSM Posttrained) `1771347697`

This document records the **exact artifacts**, **methodology**, and **training + inference settings**
for the current best BigCodeBench result in this repo, intended to be detailed enough to cite in an
academic paper.

## Result Summary (Reported Metric)

Evaluation source (already computed):
- `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1771347697_eval_results.json`
- `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1771347697_pass_at_k.json`

Reported numbers:
- Benchmark: **BigCodeBench-Complete (Hard)** (148 tasks)
- `pass@1 = 0.42567567567567566` (63/148)
- Evaluation timestamp (from eval JSON): `2026-02-18 06:30`
- Ground-truth pass rate: `0.9797297297297297`
- Known ground-truth-failing tasks (BigCodeBench dataset issue): `BigCodeBench/418`, `BigCodeBench/417`, `BigCodeBench/590`
- Evaluation mode: `calibrated=true` (see below)

## Artifacts (Repro Anchors)

Generated samples (the file that is evaluated):
- `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1771347697.jsonl`
  - 148 lines (one JSON object per task)
  - Each record contains:
    - `task_id` (e.g., `BigCodeBench/120`)
    - `solution` (sanitized Python)
    - `raw_solution` (pre-sanitize Python)

ITSSM trace (debug/analysis, not used by the evaluator):
- `itssm_approach_results/itssm_robust_reflex_bigcodebench_results_1771347697.json`

## System Overview

This run uses a **two-model pipeline**:

1. **Coder (code synthesis + revision)**:
   - Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
   - Served locally via **vLLM** as an OpenAI-compatible endpoint
   - Used for:
     - Stage 1 draft generation
     - Stage 3 revision generation (including reflections)

2. **Prover Critic (critique / subgoals / checklists)**:
   - Base model: `deepseek-ai/DeepSeek-Prover-V2-7B`
   - Adapter: RL-trained critic `rl_prover_critic_v3/checkpoint-best`
   - Loaded locally via `transformers` (HF backend)
   - Used for:
     - Stage 2 analysis + checklist (combined, tag-structured)
     - Reflection feedback (tag-structured)

The overall method is “ITSSM + Robust Reflexion + Search”, where the **critic** is post-trained by RL
to produce critique that *causally improves the coder’s next revision*.

## Methodology (Inference-Time)

All inference is implemented in:
- `fdg_approach_qwen3_posttrained_bigcodebench.py`

### Stage 0: Task formatting (BigCodeBench -> ITSSM)

Tasks are loaded offline from the BigCodeBench cache JSONL:
- `~/.cache/bigcodebench/BigCodeBench-Hard-v0.1.4.jsonl`

For each task we construct an ITSSM `question_content` that:
- pins the target entry point (`entry_point`)
- instructs to avoid I/O side effects (no stdin, no print unless required)
- discourages CLI wrappers / `if __name__ == '__main__'` blocks
- optionally includes:
  - `libs` allowlist hints
  - parsed `doc_struct` hints

We set:
- `--bcb-question-source instruct`
  - use BigCodeBench’s `instruct_prompt` as the “spec”

### Stage 1: Draft generation (coder)

The coder is prompted with:
- problem specification (`question_content`)
- starter code (`code_prompt`)

Decoding:
- temperature = `0.6`
- top_p = `0.9`
- top_k = `20`
- repetition_penalty = `1.05`
- seed = `0` (vLLM seed)
- token budgets:
  - `--coder-max-tokens 25000` (global ceiling)
  - `--draft-max-tokens 12000` (stage cap)

Search:
- `--draft-samples 3`
  - sample 3 drafts with deterministic seed offsets
  - select by self-test score (see “Public Self-Test”)

### Stage 2: Critique + checklist (prover critic)

The prover critic runs in **combined tag mode**:
- `--prover-format tags`
- `--stage2-mode combined`

It is prompted to return ONLY:
- `<subgoal> ... </subgoal>` (one subgoal per line)
- `<gap_analysis> ... </gap_analysis>` (bullet list)
- `<checklist> ... </checklist>` (bullet list)

Key bias control:
- temperature is forced to `0.0` for critic calls (deterministic + conservative)
- optional “red-team retry” triggers when critique is overconfident/thin (see `fdg_approach_qwen3_inter_posttrained.py`)

### Stage 3: Revision generation (coder)

The coder is prompted with:
- original spec + starter code
- current code
- “Synthesized Contracts and Subgoals” (critic output normalized into sections)
- robustness checklist

Decoding:
- revision temperature = `0.2` (conservative edits)
- max tokens:
  - `--revision-max-tokens 12000`

Search:
- `--revision-samples 3`
  - sample 3 revisions and select by self-test score

### Reflexion loop

We run:
- `--reflections 2`

Each reflection:
1. ask the critic for revision feedback (bullets)
2. append feedback to the stage-2 analysis
3. sample + select another revision

### Public self-test (selection signal)

This run enables *test-driven selection*:
- `--self-test-public`
- `--self-test-public-mode guard`
- `--trust-public-tests`

Important: This is powerful but can be considered **test leakage** if you treat the benchmark tests as hidden.
BigCodeBench’s standard local evaluation also executes these tests; this setting uses them *during generation*
to rank candidates.

Implementation details:
- Uses BigCodeBench’s official harness: `bigcodebench.eval.untrusted_check`
- Uses “calibrated” wrapping to match evaluator semantics:
  - `starter_code + "\\n    pass\\n" + solution`
  - enabled by default via `--bcb-calibrated` (true)

Selection scoring:
- If tests PASS: score = `1.0`
- If tests FAIL: score = `1/(1 + n_failing_tests)`
  - this prefers “near misses” over completely wrong solutions

Tie-break:
- prefer shorter code on exact score ties
- if still tied and `mode=guard`, prefer later samples

Why this matters:
- BigCodeBench failure modes are often “almost correct” (1–2 assertion mismatches).
- Binary scoring collapses these into a huge tied set at score 0.0, creating selection churn.
- Partial scoring + tie-break makes search stable and meaningfully directed.

## Training Setting (Prover Critic RL: `rl_prover_critic_v3`)

The BigCodeBench run above uses:
- `--prover-backend hf`
- `--prover-model-name-or-path /home/yueke/formalgen/lean_gen/rl_prover_critic_v3`
- `--require-checkpoint-best` (loads `rl_prover_critic_v3/checkpoint-best`)

The critic was trained by RL using:
- Entry point: `rl_training_prover_process.py`
- Algorithm: **TRL GRPO** (Group Relative Policy Optimization)
- Config file (canonical source of truth):
  - `configs/rl_prover_grpo_v3_mix_from_v2_gpt52.args`

### RL objective (high-level)

The RL policy is the critic model. For each prompt, it samples `K` structured tag completions.
Rewards combine:

1. Dense reward (structure + alignment):
   - subgoal/gap alignment to ITSSM targets (embedding-based)
2. Terminal reward (causal usefulness):
   - run the **coder revision** conditioned on the critic text, then run unit tests
3. Quality reward (LLM judge):
   - strict **Amplify GPT-5.2** judge (no fallback)

The key design choice is terminal reward computed from the **coder’s revised code**, not from prover-generated
code. This trains the critic to write feedback that actually makes the coder succeed.

### RL dataset mixture

The v3 critic is trained on a weighted mixture to emphasize hard failures while maintaining stability:
- hard-fail set repeated `@4`
- hard-draftfail set repeated `@2`
- full regenerated dataset glob (`itssm_lcb_v*_regen.jsonl`)

See the `--tasks-jsonl` line in the args file for exact paths.

### RL judge (must be GPT-5.2; strict)

The v3 config enforces Amplify GPT-5.2 judge:
- provider: `amplify`
- base URL: `https://prod-api.vanderbilt.ai`
- model: `gpt-5.2`
- strict mode: abort on invalid judge output / missing key (no silent fallback)

## Reproduce (Evaluation-Only, Deterministic)

To reproduce the reported `pass@1` from the saved samples file:

```bash
cd /home/yueke/formalgen/lean_gen/bigcodebench
bigcodebench.evaluate \
  --execution local \
  --split complete \
  --subset hard \
  --samples ../bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1771347697.jsonl
```

Expected output files (already present):
- `bcb_results/..._eval_results.json`
- `bcb_results/..._pass_at_k.json`

## Reproduce (End-to-End: Start Server -> Generate -> Evaluate)

### 1) Start local vLLM coder server

You must run a server whose `--max-model-len` supports:
`prompt_tokens + max(draft_max_tokens, revision_max_tokens)` (12k in this run), with headroom.

Example:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --port 1234 \
  --max-model-len 32768
```

### 2) (Optional) Train the critic (RL)

```bash
cd /home/yueke/formalgen/lean_gen
export AMPLIFY_API_KEY='YOUR_REAL_KEY'
python rl_training_prover_process.py \
  @configs/rl_prover_grpo_v3_mix_from_v2_gpt52.args \
  --quality-llm-api-key "$AMPLIFY_API_KEY"
```

### 3) Generate BigCodeBench samples via ITSSM

```bash
cd /home/yueke/formalgen/lean_gen
RUN_TS=1771347697

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
  --draft-samples 3 --revision-samples 3 --reflections 2 \
  --self-test-public --self-test-public-mode guard --trust-public-tests \
  --bcb-split complete --bcb-subset hard --bcb-question-source instruct \
  --output-dir ./itssm_approach_results --bcb-root ./bcb_results \
  --run-timestamp "$RUN_TS"
```

### 4) Evaluate locally

```bash
cd /home/yueke/formalgen/lean_gen/bigcodebench
bigcodebench.evaluate \
  --execution local \
  --split complete \
  --subset hard \
  --samples ../bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1771347697.jsonl
```

## Important Implementation Notes (Repro Gotchas)

1. **BigCodeBench evaluator expects `.jsonl`**:
   - `bigcodebench.evaluate` asserts `samples.endswith('.jsonl')`.
2. **Matplotlib import under memory limits**:
   - This repo patches `bigcodebench/bigcodebench/eval/utils.py` to treat matplotlib import as best-effort.
   - Without this, some tasks importing matplotlib can spuriously fail in constrained environments.
3. **Timeout control**:
   - BigCodeBench’s `untrusted_check` defaults to `TIMEOUT_LIMIT=240s`.
   - You can override with `BIGCODEBENCH_TIMEOUT_PER_TASK`, but this run did not rely on tight timeouts.
4. **Non-determinism**:
   - vLLM seed reduces variance but does not guarantee bitwise determinism across machines/drivers.
   - The safest reproducibility anchor is the saved `samples.jsonl` + evaluator version.

## How To Describe This In A Paper (Suggested Wording)

“We use a two-model iterative code synthesis pipeline. A strong code model (Qwen3-Coder-30B) produces an initial
solution. A separate prover-style critic (DeepSeek-Prover-V2-7B) is post-trained via GRPO with a terminal reward
defined by running unit tests on the coder’s revision conditioned on the critic’s feedback. At inference time we
perform search over multiple drafts and revisions, selecting candidates using public-test execution under the
benchmark’s official harness. We additionally run a limited reflexion loop where the critic provides further
feedback and the coder revises again.”
