# ITSSM Best Run `1771167978` (LiveCodeBench v6): End-to-End Methodology

This README documents the current best LiveCodeBench `v6` score in this repo and, more importantly,
how the full pipeline works from:

1. Data preprocessing (ITSSM-derived JSONL targets)
2. SFT warm-start (critic format learning)
3. RL (GRPO, process-aligned critic learning)
4. Inference (ITSSM robust-reflex with search + optional public self-test)

The best run is produced by **only**:

- Inference runner: `fdg_approach_qwen3_inter_posttrained.py`
- RL-trained prover critic: `rl_prover_critic_v3` (adapter over `deepseek-ai/DeepSeek-Prover-V2-7B`)

## Result and Artifacts

Saved evaluation metric:

- `pass@1 = 0.3942857142857143` (69/175)
  - `itssm_approach_results/itssm_robust_reflex_livebench_1771167978_codegeneration_output_eval.json`

## Critic Ablation Snapshot

Same ITSSM inference/search settings on LiveCodeBench `v6`:

| Critic | Pass/Total | pass@1 | Eval artifact |
|---|---:|---:|---|
| SFT-only (`sft_prover_critic_v2`) | 62/175 | 35.4% | `itssm_approach_results_sft_ablation/itssm_robust_reflex_livebench_1773352543_codegeneration_output_eval.json` |
| RL best (`rl_prover_critic_v3/checkpoint-best`) | 69/175 | 39.4% | `itssm_approach_results/itssm_robust_reflex_livebench_1771167978_codegeneration_output_eval.json` |

Artifacts produced by the inference script:

- Generation output (LiveCodeBench format):
  - `itssm_approach_results/itssm_robust_reflex_livebench_1771167978.json`
- Full trace (draft, critique, revisions):
  - `itssm_approach_results/itssm_robust_reflex_results_1771167978.json`

## Quick Reproduction (Best Run)

### 0) Prereqs and Safety

- Repo root: `/home/yueke/formalgen/lean_gen`
- Local vLLM OpenAI-compatible server for the **coder** at `http://localhost:1234/v1`
- GPU VRAM for:
  - Qwen coder in vLLM
  - DeepSeek Prover critic loaded by `transformers` (inference) and TRL/PEFT (training)

Safety:

- Public-test execution runs generated code. The runner uses a subprocess with timeouts, but this is not
  a hardened sandbox. Treat all code as untrusted.

### 1) Start the coder (local vLLM)

Example (adjust model and max context to your environment):

```bash
cd /home/yueke/formalgen/lean_gen
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --port 1234 \
  --max-model-len 32768
```

### 2) Regenerate the best ITSSM outputs (`1771167978`)

This is the exact command that produced the `1771167978` artifacts.

```bash
cd /home/yueke/formalgen/lean_gen
RUN_TS=1771167978

python fdg_approach_qwen3_inter_posttrained.py \
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
  --version v6 --start-index 0 --end-index 175 \
  --output-dir ./itssm_approach_results --run-timestamp "$RUN_TS"
```

### 3) Re-run LiveCodeBench evaluation

```bash
cd /home/yueke/formalgen/lean_gen
python -m lcb_runner.runner.custom_evaluator \
  --custom_output_file itssm_approach_results/itssm_robust_reflex_livebench_1771167978.json \
  --scenario code_generation \
  --release_version v6 \
  --output_dir itssm_approach_results
```

Expected:

- `itssm_approach_results/itssm_robust_reflex_livebench_1771167978_codegeneration_output_eval.json`
  contains `pass@1 = 0.3942857142857143`.

## System Overview

### Components

At runtime, we use a two-model system:

1. **Coder**: Qwen3 Coder, served via vLLM (OpenAI-compatible `chat.completions` API).
   - Writes and revises Python solutions.
2. **Prover/Critic**: DeepSeek Prover V2 7B + PEFT adapter (trained in this repo).
   - Does not write code at inference time.
   - Emits structured feedback tags that steer the coder.
3. **Public-test executor**: LiveCodeBench runner invoked via `LiveCodeBenchExecutor` (subprocess).

### ITSSM robust-reflex loop (baseline vs posttrained)

Baseline ITSSM robust-reflex pipeline:

- Script: `fdg_approach_qwen3_inter_refined_reflex_robust.py`
- Stages:
  1. Stage 1: coder drafts code
  2. Stage 2a: prover "type analysis" (subgoals + gap analysis)
  3. Stage 2b: prover "robustness checklist"
  4. Stage 3: coder revision using (analysis + checklist)
  5. Optional reflexion loop: extra feedback -> revise again

Posttrained pipeline (used for this best run):

- Script: `fdg_approach_qwen3_inter_posttrained.py`
- Keeps the same revision prompt structure, but adds:
  - HF-loaded prover critic (`--prover-backend hf`)
  - tag-normalization and a single combined Stage-2 call (aligned with RL training)
  - multi-sample search + optional public-test selection

## Inference Methodology (Search + Public Self-Test)

Script: `fdg_approach_qwen3_inter_posttrained.py`

Implementation notes (what actually happens):

- Prover format:
  - `--prover-format tags --stage2-mode combined` makes Stage-2 a single call that returns only:
    `<subgoal>...</subgoal><gap_analysis>...</gap_analysis><checklist>...</checklist>`
  - The script converts these tags back into the numbered "type_analysis" style that the coder revision
    prompt expects (this keeps the coder prompt stable across versions).

- Search:
  - `--draft-samples N`: sample N independent drafts (Stage 1).
  - `--revision-samples N`: sample N independent revisions per critique (Stage 3).
  - Sampling uses the coder seed as a base and offsets it per candidate when possible.

- Public self-test selection (optional, enabled in the best run):
  - `--self-test-public` runs LiveCodeBench public tests during generation.
  - `--self-test-public-mode`:
    - `select`: keep max public score; ties keep earlier candidate.
    - `guard`: keep max public score; ties prefer later candidate (often the "fix" iteration).
    - `hint`: do not select, only use test outcomes as a hint for red-team critique behavior.
  - `--no-trust-public-tests`: perfect public score is not treated as "correct"; the critic remains paranoid.

- Red-team retries on the critic (stability):
  - If the critic output looks overconfident or too thin (few bullets, "no issues"), the script retries
    once with a stricter "assume wrong" prompt to get actionable gaps.

Why the best-run decoding settings matter:

- Low revision temperature (`--revision-temperature 0.2`): revisions should be conservative and actually
  apply the critique, not rewrite the solution randomly.
- Large overall token budget with stage caps:
  - `--coder-max-tokens 25000` allows long prompts and reasoning.
  - `--draft-max-tokens 12000 --revision-max-tokens 12000` caps output bloat to keep critique effective.

## Data Preprocessing Methodology (ITSSM JSONL Targets)

The critic is trained against targets generated by running an ITSSM pipeline on LiveCodeBench tasks.

### Why we do ITSSM-derived preprocessing (and not "raw LCB -> RL" directly)

We need three properties simultaneously:

1. **Process alignment**: the prover is used as a *critic* that conditions a *separate coder* (Qwen).
   Training data therefore must contain both the *problem* and a realistic *draft_code* for the critic to analyze.
2. **A dense learning signal**: public-test pass/fail is sparse and noisy, especially when:
   - drafts are far from correct,
   - public tests are few/weak,
   - the critic influences correctness only indirectly through the coder.
3. **Interface stability**: the coder prompt structure is intentionally kept stable across experiments.
   We want to improve the system by training the critic, without constantly retuning/retraining the coder.

The ITSSM-derived JSONL preprocessing provides:

- `draft_code`: what the coder actually wrote for the task under the same prompting regime used at evaluation.
- `subgoals/gap_analysis/robustness_checklist`: supervision targets that define "what a good critique looks like" in
  the *exact* style that helps the coder revise.

This makes both SFT and RL significantly more stable than relying only on sparse test reward.

### LiveCodeBench task loading (offline-friendly)

`train_deepseek_prover_rl.py` provides `_load_tasks_from_livecodebench(...)`:

- Reads cached HuggingFace datasets Arrow shards from `~/.cache/huggingface/datasets`
- Avoids dataset locking/writes by directly loading Arrow files
- Returns task dicts containing:
  - `question_id`, `question_content`, `starter_code`
  - `input_output` JSON used by the executor

Implementation details worth knowing:

- The offline loader uses the cached `release_latest` dataset as the source of truth and then derives contiguous
  slices for `v1..v6` in that cache order. (It also supports tags like `release_v5` and `release_latest`.)
- `input_output` is a JSON payload used by LiveCodeBench's own runner. It typically contains:
  - `inputs`: list[str] (each string is raw stdin)
  - `outputs`: list[str] (raw expected stdout)
  - `fn_name`: function name for LeetCode-style tasks

By default, we use **public tests**. Some scripts accept `--include-private-tests`, but this should be treated as an
optional knob for internal experiments, not a default assumption.

### Dataset generation script

Script: `generate_itssm_rl_dataset_livecodebench.py`

For each task:

1. Coder drafts (`draft_code`) using the same Qwen prompt structure as ITSSM inference.
2. Prover produces analysis and checklist:
   - `--prover-backend deepseek`: uses `DeepSeekReasonerClient` (OpenAI-compatible DeepSeek endpoint)
   - `--prover-backend vllm`: uses a local OpenAI-compatible prover endpoint
3. Coder revises (`revised_code`) using the analysis/checklist.
4. Extract targets:
   - `subgoals` (list[str]) and `gap_analysis` (str) from `type_analysis`
   - `robustness_checklist` (str), with fallbacks if missing
5. Write a JSONL record.

#### Coder prompting and cleanup (what ends up in `draft_code` / `revised_code`)

The dataset generator reuses the ITSSM prompt structure from `fdg_approach_qwen3_inter_refined_reflex_robust.py`:

- Stage 1 draft prompt: Qwen-style "write the full solution, return only a Python code block".
- Stage 3 revision prompt: same structure used at inference:
  - includes the original problem + starter code + the current code
  - includes the synthesized "contracts/subgoals" text (analysis + checklist)
  - instructs "Return ONLY the corrected code in a single Python code block"

After generation, we normalize code:

- strip fenced code blocks
- remove any `if __name__ == "__main__": ...` blocks (conservative regex)

These cleanup steps matter because:

- LiveCodeBench expects a pure solution class/function, not a script.
- Extra text/code blocks in the output can cause test runner errors and poison training labels.

#### Target extraction + quality control (how `subgoals/gap/checklist` are created)

The teacher prover output is not assumed to be perfectly formatted. The generator applies best-effort parsing:

- `type_analysis` is expected to contain (loosely) the three sections:
  1. Preconditions/postconditions
  2. Key invariants / subgoals
  3. Concise gap analysis
- `extract_subgoals_and_gap_analysis(...)` handles common variants:
  - numbered headings (`2.` / `3.`)
  - markdown headings (`## Key invariants / subgoals`)
  - "Subgoals:" subheadings

Checklist robustness:

- If the checklist is missing from the combined analysis, the script:
  1. retries checklist generation once (best-effort) using the same prover backend, and if still empty
  2. falls back to a heuristic checklist derived from the gap text

This matters because both SFT and RL filter on "tag completeness" (via `--min-present-target-tags`).
If you do not fill missing checklists, you end up dropping a large fraction of rows or training on incomplete tags.

#### Optional test labeling (auditing + hard-slice construction)

With `--run-tests`, the generator executes `revised_code` on LiveCodeBench tests and stores:

- `passed` (bool)
- `test_status`
- `test_details` (includes per-test results + `meta` like WA/RE/TLE error codes)

We use this for two reasons:

1. **Audit**: verify the teacher pipeline isn't emitting obviously broken or non-executable code too often.
2. **Curriculum**: build focused "hard fail" slices (e.g. v6 tasks the system still fails) to upweight during RL.
   In this repo those curated slices are stored under `itssm_rl_dataset_artifacts/` and referenced by RL configs.

Record schema (typical):

```json
{
  "question_id": "v1/1873_A",
  "lcb_version": "v1",
  "question_title": "...",
  "question_content": "...",
  "starter_code": "...",
  "input_output": "{\"inputs\":[...],\"outputs\":[...],\"fn_name\":\"...\"}",
  "metadata": {...},
  "draft_code": "...",
  "type_analysis": "...",
  "robustness_checklist": "...",
  "subgoals": ["...", "..."],
  "gap_analysis": "...",
  "revised_code": "..."
}
```

Optional test labeling:

- If `--run-tests` is set, the script executes `revised_code` on public tests and stores
  `passed/test_status/test_details`. This is useful to create focused "hard fail" datasets.

Example regeneration command:

```bash
cd /home/yueke/formalgen/lean_gen
python generate_itssm_rl_dataset_livecodebench.py \
  --lcb-versions v1 \
  --limit 999999 --start-index 0 --end-index 999999 \
  --prover-backend deepseek \
  --qwen-base-url http://localhost:1234/v1 \
  --output-jsonl itssm_lcb_v1_regen.jsonl
```

### What preprocessing outputs we actually used (v3 critic training)

The v3 critic (`rl_prover_critic_v3`) is trained on a *mixture* of preprocessed JSONLs:

Broad "regen" coverage (ITSSM-derived, used for stability/generalization):

- `itssm_lcb_v1_regen.jsonl`
- `itssm_lcb_v2_regen.jsonl`
- `itssm_lcb_v3_regen.jsonl`

Focused v6 slices (curriculum/upweighting, used to target known failures):

- `itssm_rl_dataset_artifacts/itssm_lcb_v6_hard_fail_1770671000.jsonl`
- `itssm_rl_dataset_artifacts/itssm_lcb_v6_hard_draftfail_1770671000.jsonl`

Justification for mixing (instead of training only on the hard slice):

- The hard slice is high-signal but small; training solely on it tends to overfit and can degrade general critique quality.
- The broad regen set keeps the critic competent on "normal" problems and prevents the model from learning a narrow,
  adversarial style tuned only to a particular failure pattern.
- Upweighting via `@N` repeats in the RL config lets us bias the gradient toward hard cases without collapsing coverage.

The exact mixture weights are in `configs/rl_prover_grpo_v3_mix_from_v2_gpt52.args` (see `--tasks-jsonl`).

## SFT Warm-Start Methodology (Critic Format Learning)

Script: `rl_training_prover_sft.py`

### Why SFT before RL (and why we keep SFT narrow)

GRPO RL here is high-variance for two structural reasons:

1. The policy output (critic text) only affects correctness **indirectly** through a separate model (the coder).
2. The terminal reward is computed via code execution, which is:
   - expensive,
   - noisy (timeouts, flaky failures),
   - and often sparse on hard tasks.

SFT is used to cheaply enforce the "interface contract" of the critic before RL:

- output must be parseable tags (no prose around it)
- output must not contain code (critic role, not solver role)
- output should be structured enough that the coder can reliably consume it

We keep SFT narrow on purpose:

- It teaches **format + role + basic content mapping** from ITSSM targets.
- It does not try to solve tasks or produce final code.
- RL then refines the critic based on downstream coder success.

In practice, SFT also acts as a "schema validator" for the preprocessing pipeline:

- If SFT loss does not decrease or the model keeps emitting malformed tags, it is usually because:
  - `draft_code` is too long/noisy (truncation removes the key bug),
  - the targets are empty/low-quality (parsing failed during preprocessing),
  - or the prompt template drifted (mismatch between preprocessing and training).

What it learns:

- Input: (problem statement + draft code)
- Output: strictly the Stage-2 combined tag format:
  - `<subgoal>`: one per line
  - `<gap_analysis>`: bullets
  - `<checklist>`: bullets

Key implementation details:

- Prompt format is identical to RL/inference:
  - `build_prover_stage2_prompt(problem, draft_code)` (from `rl_training_prover_process.py`)
- Targets are extracted with robust fallbacks:
  - `extract_itssm_targets(...)` (from `rl_training_prover_process.py`)
    - derives missing `subgoals/gap` from `type_analysis` for legacy datasets
- Filtering:
  - drops rows with fewer than `--min-present-target-tags` non-empty targets
- Supervision:
  - prompt tokens are masked (`labels=-100`), so the loss is only on the completion tokens
- Output:
  - a PEFT adapter directory with `base_model.json` (so downstream scripts can auto-resolve the base model)

### Prompt/completion format (exactly what the model sees/produces)

Prompt (from `build_prover_stage2_prompt(...)` in `rl_training_prover_process.py`):

```text
You are DeepSeek Prover. Do NOT write any code.
Given the problem description and the candidate implementation, identify key subgoals, a concise gap analysis,
and a robustness checklist for a coder to fix the program.

Problem Description:
{problem}

Candidate Implementation (Python):
```python
{draft_code}
```

Return ONLY these tags:
<subgoal>one subgoal per line</subgoal>
<gap_analysis>bullet list</gap_analysis>
<checklist>bullet list</checklist>
```

Completion format (what SFT supervises):

```text
<subgoal>
... one subgoal per line ...
</subgoal>
<gap_analysis>
- ...
- ...
</gap_analysis>
<checklist>
- ...
- ...
</checklist>
```

The tags are intentionally minimal and rigid. This makes the critic output:

- easy to parse reliably
- stable across model/provider differences
- easy to feed into the coder revision prompt with low prompt-engineering overhead

### Filtering and truncation (why it matters)

SFT drops low-quality rows by requiring `--min-present-target-tags` non-empty targets among:

- `subgoals`
- `gap_analysis`
- `robustness_checklist`

Justification:

- Missing targets imply the teacher output was malformed or unparseable for that row.
- Keeping such rows teaches the model to emit empty tags, which RL then has to unlearn.

Token limits:

- prompt truncation: `--max-prompt-tokens` (default 1536)
- completion truncation: `--max-completion-tokens` (default 512)

Justification:

- For a critic, the most important information is usually in the problem statement and the top portion of the draft.
  Beyond a point, longer drafts mostly add noise and cause OOMs.
- Keeping completions capped prevents the model from producing long essays and forces concise actionable critiques.

If you observe frequent truncation, prefer:

1. increasing `--max-prompt-tokens` modestly (VRAM permitting), and
2. regenerating drafts to be shorter/cleaner (so the critic sees signal, not bloat),

before increasing completion length substantially.

### Why QLoRA + output artifacts

We use PEFT (typically QLoRA) for SFT for practical reasons:

- DeepSeek Prover V2 7B full finetuning is expensive; QLoRA fits comfortably on 48GB-class GPUs.
- The critic task is format/role adaptation plus modest content steering, which is well-suited to adapters.

We save a `base_model.json` file into the adapter directory so inference scripts can load the correct base model
without manual bookkeeping. This is required by `HFProverClient` in `fdg_approach_qwen3_inter_posttrained.py`
when the adapter directory does not include a full tokenizer config.

Example SFT command:

```bash
cd /home/yueke/formalgen/lean_gen
python rl_training_prover_sft.py \
  --model-name-or-path deepseek-ai/DeepSeek-Prover-V2-7B \
  --tasks-jsonl 'itssm_lcb_v*_regen.jsonl' \
  --peft qlora --gradient-checkpointing \
  --max-steps 120 \
  --output-dir sft_prover_critic
```

## RL Methodology (GRPO, Process-Aligned Critic)

Script: `rl_training_prover_process.py`

This is the RL loop that produces `rl_prover_critic_v3`.

### What is optimized (RL framing)

- State/context: `build_prover_stage2_prompt(problem, draft_code)`
- Action: the critic completion (combined tags only)
- Environment rollout for reward:
  1. parse the tags
  2. construct a coder revision prompt from the critique
  3. call the coder to produce revised code
  4. execute public tests on the revised code

### Reward design

For each completion:

```
reward = dense_weight * dense + terminal_weight * terminal
```

Dense term:

1. Embedding alignment to ITSSM targets (default scorer):
   - `EmbeddingCosineScorer` from `train_deepseek_prover_rl.py`
   - embeds each generated string and each target string and computes cosine similarity mapped to `[0, 1]`
2. Critique quality:
   - either a heuristic `critique_quality_score(...)`, or
   - an LLM judge (`LLMJudge`) that returns a JSON `{"score": ...}` in `[0, 1]`

Terminal term:

- Uses `LiveCodeBenchExecutor` (subprocess runner) from `train_deepseek_prover_rl.py`
- Common configuration (used in v3):
  - `--terminal-mode coder`: terminal score from coder revision result
  - `--terminal-relative --terminal-baseline draft`: terminal = score(with_critique) - score(draft_only)
  - `--skip-baseline-perfect`: drop tasks where the baseline already passes
  - `--terminal-score shaped`: shaped score avoids all-zero batches
  - `--terminal-runtime-penalty`: small penalty per rollout

#### Dense reward: implementation-level details

Dense reward is computed from the parsed tags plus the JSONL targets. In `rl_training_prover_process.py`:

- The completion is parsed into three text blocks (code emission is discouraged/penalized):
  - `gen_subgoals`: list of subgoal lines from `<subgoal>...</subgoal>`
  - `gen_gap`: string from `<gap_analysis>...</gap_analysis>`
  - `gen_check`: string from `<checklist>...</checklist>`
- Targets are read from the training row using `extract_itssm_targets(...)`, which:
  - prefers explicit `subgoals/gap_analysis/robustness_checklist`
  - falls back to parsing from `type_analysis` if needed (for legacy JSONLs)

Embedding alignment uses the scorer (default: `EmbeddingCosineScorer`) as follows:

- Subgoals: average similarity over paired lines:
  - `dense_sub = mean(score(gen_subgoals[j], target_subgoals[j]))` for `j < min(len(gen), len(target))`
- Gap and checklist: score the whole block text:
  - `dense_gap = score(gen_gap, target_gap)`
  - `dense_check = score(gen_check, target_checklist)`

These are combined into an embedding-alignment subtotal:

```
embed_dense =
  dense_subgoal_weight   * dense_sub +
  dense_gap_weight       * dense_gap +
  dense_checklist_weight * dense_check
```

Critique quality is either:

- `critique_quality_score(...)`: a heuristic in `[0, 1]` that rewards structure, coverage, grounding to identifiers
  from the draft code, and penalizes code emission in the completion, or
- `LLMJudge.score(...)`: a frozen "LLM-as-a-judge" score in `[0, 1]` (strict JSON parsing when enabled).

Final dense value:

```
dense = dense_embed_weight * embed_dense + dense_quality_weight * quality
```

No-signal handling (important for stability with relative terminal reward):

- If `--terminal-relative` is enabled and both `score_with` and `score_base` are approximately zero,
  `dense` is scaled by `--dense-no-terminal-scale`.
- In v3, this is set to `0.0` to avoid learning purely from proxies on tasks where public tests provide no gradient.

#### Terminal reward: implementation-level details

Terminal reward is computed by executing public tests via `LiveCodeBenchExecutor`:

- It runs `lcb_runner.evaluation.testing_util.run_test` inside a `multiprocessing.Process` with a timeout,
  and returns `TestResult(details={"results": [...], "meta": {"error_code": ...}})`.
- Typical error codes seen in the shaped scoring path:
  - WA: `-2`
  - TLE: `-3`
  - RE: `-4`

Terminal score has three modes (`--terminal-score`):

- `binary`: `1.0` if all tests pass, else `0.0`
- `fraction`: fraction of public tests passed (when available)
- `shaped` (used in v3): combines fraction + failure-type shaping + tiny output similarity for WA:
  - `score = 0.85 * frac + base + 0.1 * sim`
  - `base = 0.05` for WA, `0.02` for TLE, `0.0` for RE
  - `sim` is `SequenceMatcher(output, expected)` (only used for WA)

Terminal reward is then:

- With `--terminal-mode coder` (used in v3): run a coder revision call conditioned on the critic text,
  execute tests on revised code, and compute `score_with`.
- With `--terminal-relative --terminal-baseline draft` (used in v3):
  - `terminal = score_with - score_base`, where `score_base` is computed by testing `draft_code` directly.
- RE penalty:
  - if the revised code run crashes (RE, `error_code=-4`), subtract `--terminal-runtime-penalty`.
- Optional absolute shaping:
  - `terminal += terminal_abs_weight * score_with` (v3 sets this to 0.0)
- Baseline-perfect gating:
  - with `--skip-baseline-perfect`, rows with `score_base >= baseline_perfect_threshold` get total reward forced to 0.

#### GRPO mechanics (what TRL is doing)

`rl_training_prover_process.py` uses `trl.GRPOTrainer`:

- For each prompt, sample `K = --num-generations` completions from the policy.
- Compute rewards for each completion, then compute group-relative advantages internally (no value model).
- Apply a KL regularization term against the reference policy (strength `--beta`) for stability.

Sampling knobs that affect exploration and VRAM:

- `--num-generations`: group size per prompt
- `--generation-batch-size`: how many completions are generated per forward pass (often the #1 VRAM knob)
- `--temperature` / `--top-p`: policy sampling distribution during RL

#### v3 hyperparameters (from the args file)

The canonical source of truth is `configs/rl_prover_grpo_v3_mix_from_v2_gpt52.args`. Key values:

- Data: `itssm_lcb_v*_regen.jsonl` plus upweighted v6 hard slices using `@N` repeats
- PEFT: `--peft qlora --gradient-checkpointing`
- GRPO: `--num-generations 3 --generation-batch-size 2 --max-steps 120`
- Reward weights: `--dense-weight 0.2 --terminal-weight 0.8`
- Terminal: `--terminal-mode coder --terminal-relative --terminal-baseline draft --terminal-score shaped`
- Judge: `--quality-reward llm --quality-llm-provider amplify --quality-llm-model gpt-5.2 --quality-llm-strict`
- Coder inside terminal reward: conservative decoding (`--coder-temperature 0.2 --coder-max-tokens 12000 --coder-seed 0`)

### Strict GPT-5.2 judge (v3)

The v3 config enforces:

- provider: Amplify
- base URL: `https://prod-api.vanderbilt.ai`
- model: `gpt-5.2`
- strict parsing: any judge output that is invalid/unparsable aborts training

This is deliberate to avoid silent reward corruption during long RL runs.

### Dataset mixing and upweighting

The v3 config uses the `@N` repeat syntax in `--tasks-jsonl` to upweight hard slices.

See:

- `configs/rl_prover_grpo_v3_mix_from_v2_gpt52.args`

### Checkpointing

`rl_training_prover_process.py` supports:

- periodic checkpoints (`checkpoint-*`)
- `checkpoint-best` tracked by a chosen metric (default: reward)

The best-run inference uses:

- `--require-checkpoint-best` (refuses to use a regressed late checkpoint)

### Exact v3 training command

Start the coder server (used inside terminal reward):

```bash
cd /home/yueke/formalgen/lean_gen
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --port 1234
```

Run GRPO with the strict judge:

```bash
cd /home/yueke/formalgen/lean_gen
export AMPLIFY_API_KEY='YOUR_REAL_KEY'
python rl_training_prover_process.py \
  @configs/rl_prover_grpo_v3_mix_from_v2_gpt52.args \
  --quality-llm-api-key "$AMPLIFY_API_KEY"
```

Outputs:

- `rl_prover_critic_v3/checkpoint-best/` (recommended for inference)
- `rl_prover_critic_v3/checkpoint_summary.json` (best step, saved periodic steps, etc)

## Common Pitfalls

1. vLLM context overflow:
   - Fix by increasing vLLM `--max-model-len` or lowering `--draft-max-tokens/--revision-max-tokens`.
2. Missing `checkpoint-best`:
   - If `--require-checkpoint-best` is set, the directory must exist.
3. Judge failures abort training:
   - With `--quality-llm-strict`, invalid judge output aborts. This is intentional for reproducibility.
4. Degenerate terminal reward (all-zero):
   - Usually indicates a coder/test mismatch. The RL script can abort rather than silently training on proxies.

## Suggested Next Experiments (Beyond 69/175)

Keep the best-run command as a fixed baseline (same seed), then change one axis at a time:

1. Increase search depth without raising temperatures:
   - `--draft-samples 3 --revision-samples 3` (keep `--revision-temperature 0.2`)
2. Add one more reflexion round:
   - `--reflections 3` (watch runtime)
3. Upweight "near misses":
   - build a JSONL of tasks that pass public tests but fail hidden tests, then repeat with `@N`
   - retrain a v4 critic with the same strict judge and best-checkpoint selection
