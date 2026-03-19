## BigCodeBench Evaluation for Reflexion & Self‑Refine (Qwen3)

This README explains how to run the **Reflexion** and **Self‑Refine** Qwen3
pipelines on **BigCodeBench**, and how to evaluate the generated code using
the official `bigcodebench.evaluate` command.

The generation scripts directly write **BigCodeBench-compatible** `samples.jsonl`
files (`task_id` + `solution` + `raw_solution`), so **no extra transfer step**
is required.

---

### 1. Prerequisites

From the project root (one level above `lean_gen/`):

- BigCodeBench repo is cloned under `lean_gen/bigcodebench/` (already done).
- BigCodeBench Python package and eval requirements are installed, e.g.:

```bash
cd lean_gen/bigcodebench
pip install -e .
pip install -r Requirements/requirements-eval.txt
```

- Your Qwen3 models are accessible via an OpenAI-compatible endpoint, as used
  in the existing scripts (e.g. `QWEN_VLLM_BASE_URL`, `CRITIC_VLLM_BASE_URL`).

---

### 2. BigCodeBench tasks, splits, and sizes

BigCodeBench is a HumanEval-style benchmark with **function-level** coding
tasks, but with more complex instructions and richer library usage.

**Splits (`--bcb-split`)**

- `instruct` (default)
  - Input: natural-language instructions (`instruct_prompt`).
  - Best for instruction-tuned / chat models (like your Qwen3 settings).
- `complete`
  - Input: code + docstring completion prompts (`complete_prompt`).
  - Best for base code models that do direct completion.

**Subsets (`--bcb-subset`)**

- `full`
  - ~1.1K tasks (the full BigCodeBench), covering a wide range of practical
    software-engineering problems.
  - Use this when you want a **thorough** evaluation.
- `hard`
  - ~148 tasks (BigCodeBench-Hard), a curated subset of more challenging and
    realistic problems.
  - Much faster to run; good for **quick iterations** and ablations.

**Which configuration should I use?**

- For **primary results** with Qwen3 Reflexion / Self‑Refine:
  - `--bcb-split instruct --bcb-subset full`
  - This matches the usual “chat / instruction-following” use case.
- For **quick experiments** or when compute is limited:
  - `--bcb-split instruct --bcb-subset hard`
  - Same interface, much smaller set of tasks.
- If you specifically want to test **code-completion behavior** of a base
  coder model:
  - `--bcb-split complete --bcb-subset full` (or `hard` for a smaller run).

Our scripts always generate **one sample per task** for the chosen
`split`/`subset`. You only need to choose the combination that matches your
model type and the evaluation budget you have.

---

### 3. Scripts Overview

Two new scripts live in `lean_gen/`:

- **Reflexion + BigCodeBench**
  - `reflexion_feedback_qwen3_bigcodebench.py`
  - Uses `ReflexionApproachGenerator` from `reflexion_feedback_qwen3_improved.py`.
  - Writes BigCodeBench samples under `bcb_results/`.

- **Self‑Refine + BigCodeBench**
  - `self_refine_qwen3_bigcodebench.py`
  - Uses `SelfRefineGenerator` from `self_refine_qwen3.py`.
  - Writes BigCodeBench samples under `bcb_results/`.

Both scripts:

- Use BigCodeBench **Instruct** or **Complete** splits (`--bcb-split`).
- Support **full** or **hard** subsets (`--bcb-subset`).
- Map BigCodeBench tasks into the existing pipelines without changing any
  model or sampling settings.
- Produce one sample per task with structure:

```json
{"task_id": "BigCodeBench/...?", "solution": "<sanitized_code>", "raw_solution": "<original_code>"}
```

---

### 4. Running Reflexion on BigCodeBench

From `lean_gen/`:

```bash
python reflexion_feedback_qwen3_bigcodebench.py \
  --bcb-split instruct \
  --bcb-subset full \
  --bcb-root bcb_results \
  --output-dir reflexion_feedback_results
```

Key flags (all have sensible defaults):

- `--bcb-split {instruct,complete}`
  - `instruct` (default): uses `instruct_prompt` as the natural-language task.
  - `complete`: uses `complete_prompt` as the task description.
- `--bcb-subset {full,hard}`
  - `full` (default): full BigCodeBench.
  - `hard`: BigCodeBench-Hard subset.
- `--bcb-root`: directory for BigCodeBench samples (default `bcb_results`).
- Model / sampling args mirror the original Reflexion script, e.g.:
  - `--qwen-model`, `--qwen-base-url`
  - `--critic-model`, `--critic-base-url`
  - `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`
  - `--reflections`, `--critic-temperature`

After running, you will see a file similar to:

- `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-instruct--reflexion-0.6-1.jsonl`

Each line has BigCodeBench’s expected keys:

```json
{"task_id": "BigCodeBench/...?", "solution": "<sanitized_code>", "raw_solution": "<original_code>"}
```

---

### 5. Running Self‑Refine on BigCodeBench

From `lean_gen/`:

```bash
python self_refine_qwen3_bigcodebench.py \
  --bcb-split instruct \
  --bcb-subset full \
  --bcb-root bcb_results \
  --output-dir self_refine_results
```

Important flags:

- `--bcb-split {instruct,complete}` (default `instruct`).
- `--bcb-subset {full,hard}` (default `full`).
- `--bcb-root` for the samples output directory (default `bcb_results`).
- Model / sampling args match the original Self‑Refine script:
  - `--coder-model`, `--coder-base-url`
  - `--critic-model`, `--critic-base-url`
  - `--iterations`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`
  - `--critic-temperature`

Output file example:

- `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-instruct--selfrefine-0.6-1.jsonl`

Same JSONL schema as Reflexion: `task_id`, `solution`, `raw_solution`.

---

### 6. Evaluating with BigCodeBench

Once you have a `samples.jsonl` file from either script, you can run
BigCodeBench’s official evaluation. You can run it via **Docker** (recommended)
or directly via the Python CLI.

#### 6.1 Local evaluation via CLI

Make sure BigCodeBench eval dependencies are installed (see Prerequisites).

Example for Reflexion outputs (instruct, full):

```bash
cd lean_gen
bigcodebench.evaluate \
  --execution local \
  --split instruct \
  --subset full \
  --samples bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-instruct--reflexion-0.6-1.jsonl
```

Example for Self‑Refine outputs:

```bash
cd lean_gen
bigcodebench.evaluate \
  --execution local \
  --split instruct \
  --subset full \
  --samples bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-instruct--selfrefine-0.6-1.jsonl
```

You can adjust:

- `--split instruct|complete` to match the split used during generation.
- `--subset full|hard` to match the subset.
- Optional flags from BigCodeBench (see `ADVANCED_USAGE.md`):
  - `--pass_k 1,5,10` to customize pass@k.
  - `--no-gt` if you don’t want to re-check ground truths.
  - `--parallel` to control concurrency.

BigCodeBench will print metrics like `pass@1`, `pass@5`, and `pass@10` and
write detailed results and pass@k summaries alongside your samples.

#### 6.2 Evaluation via Docker

You can also use the official Docker image (recommended for isolation):

```bash
cd lean_gen
docker run -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest \
  --execution local \
  --split instruct \
  --subset full \
  --samples /app/bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-instruct--reflexion-0.6-1.jsonl
```

Replace the `--samples` path with the Self‑Refine file for that approach.

---

### 7. Notes and Tips

- Each script generates **one sample per task** for all tasks in the chosen
  BigCodeBench split/subset.
- For tasks where generation fails, the scripts still emit a record with a
  `solution` string starting with `# Error...`. BigCodeBench treats these as
  failed attempts but ensures every task has at least one sample, avoiding
  assertion errors during evaluation.
- The scripts call BigCodeBench’s `sanitize` helper on successful generations
  to extract the relevant code around the `entry_point`, improving comparability
  with other models on the leaderboard.
- All model hyperparameters (temperature, top‑p, etc.) are inherited from the
  original LiveCodeBench scripts; you can tune them via the CLI flags if you
  want different sampling behavior on BigCodeBench.
