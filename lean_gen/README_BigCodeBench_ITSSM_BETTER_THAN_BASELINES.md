# Why ITSSM Beats Reflexion and One-Shot on BigCodeBench (Evidence + Generalization Argument)

This document is a paper-ready justification that our ITSSM pipeline (coder + RL-trained prover-critic + search) is **better** than (1) a single-model Reflexion loop and (2) a one-shot “simple Qwen3” baseline on **BigCodeBench-Complete (Hard)**. It uses only metrics computed from artifacts already committed in this repo.

## 0) Artifacts (Anchors)

**ITSSM (ours, posttrained, run `1771347697`)**
- Generation trace (includes `initial_proof_term`, `revised_python`, and `Public test signal`): `itssm_approach_results/itssm_robust_reflex_bigcodebench_results_1771347697.json`
- Local evaluator outputs:
  - `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1771347697_eval_results.json`
  - `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1771347697_pass_at_k.json`

**Baseline A (Reflexion)**
- `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--reflexion-0.6-1_eval_results.json`
- `bcb_results/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--reflexion-0.6-1_pass_at_k.json`

**Baseline B (Simple Qwen3, one-shot)**
- `simple_results/simple_qwen3_hard_complete_v2_eval_results.json`
- `simple_results/simple_qwen3_hard_complete_v2_pass_at_k.json`

**Baseline C (ITSSM-SFT ablation, run `1773431091`)**
- `bcb_results_sft_ablation/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1773431091_eval_results.json`
- `bcb_results_sft_ablation/Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8--bigcodebench-hard-complete--itssm-posttrained-1773431091_pass_at_k.json`

**Localization analysis outputs**
- `README_BigCodeBench_LOCALIZATION_REPORT_1771347697.md`
- `reports/bcb_localization_overview.json`
- `reports/bcb_localization_per_task.csv`
- `reports/bcb_localization_edit_types_reflexion_improved.json`
- `reports/bcb_localization_edit_types_reflexion_improved.csv`
- `reports/bcb_localization_edit_types_simple_improved.json`
- `reports/bcb_localization_edit_types_simple_improved.csv`
- `reports/bcb_localization_rl_with_sft/bcb_localization_overview.json`
- `reports/bcb_localization_rl_with_sft/bcb_trigger_fix_distance.csv`
- `reports/bcb_localization_rl_with_sft/bcb_localization_edit_types_itssm_sft_improved.json`
- `reports/bcb_trigger_fix_amplify_sft_annotations.csv`
- `reports/bcb_trigger_fix_amplify_sft_summary.json`

## 1) Topline: ITSSM Improves Pass@1 (Same Model Family, Same Eval)

All numbers are **BigCodeBench-Complete (Hard)**, 148 tasks, local evaluation, `calibrated=true`.

- **ITSSM:** `pass@1 = 0.4257` (63/148)
- **Reflexion:** `pass@1 = 0.3649` (54/148; includes 1 timeout)
- **Simple:** `pass@1 = 0.3446` (51/148)

Absolute gains:
- ITSSM vs Reflexion: `+9` solved tasks (Δpass@1 `+0.0608`)
- ITSSM vs Simple: `+12` solved tasks (Δpass@1 `+0.0811`)

This is not “cherry-picked improvements”: the per-task deltas show a consistent win/loss structure:
- **Wins/losses vs Reflexion:** `16` tasks where Reflexion fails but ITSSM passes, `7` tasks where Reflexion passes but ITSSM fails (net `+9`). See `reports/bcb_localization_overview.json` `comparisons.improved_over_reflexion` and `comparisons.regressed_vs_reflexion`.
- **Wins/losses vs Simple:** `19` improved, `7` regressed (net `+12`). See `reports/bcb_localization_overview.json` `comparisons.improved_over_simple` and `comparisons.regressed_vs_simple`.
- **Wins/losses vs ITSSM-SFT:** `4` improved, `3` regressed (net `+1`). See `reports/bcb_localization_rl_with_sft/bcb_localization_overview.json` `comparisons.improved_over_itssm_sft` and `comparisons.regressed_vs_itssm_sft`.

### Table 3 (Updated)

Table 3: BigCodeBench-Complete (Hard): overall performance, win/loss decomposition, and failure-mode shift. ITSSM achieves the highest pass@1 while also reducing crash-like failures relative to all baselines, including the SFT-only critic ablation.

| Method | Solved | pass@1 | ITSSM Wins | ITSSM Regr. | Fail-Crash | Fail-AssertionOnly | Fail-Mixed | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| One-shot (coder-only) | 51/148 | 0.3446 | 19 | 7 | 30 | 53 | 14 | 0 |
| Reflexion | 54/148 | 0.3649 | 16 | 7 | 33 | 50 | 10 | 1 |
| ITSSM-SFT | 62/148 | 0.4189 | 4 | 3 | 23 | 51 | 11 | 1 |
| ITSSM (ours) | 63/148 | 0.4257 | - | - | 21 | 52 | 11 | 1 |

### Table 4 (Updated)

Table 4: BigCodeBench mechanism table: ITSSM’s wins over baselines are usually small and local. The strongest evidence appears in the Reflexion -> ITSSM comparison, where repairs are near-trigger and concentrated in thin-contract regions (exception/string/API constraints). The SFT-ablation row shows that the RL gain over SFT is smaller in count but materially less localized than the Reflexion gap.

| Baseline -> ITSSM | N Wins | Zero-Edit Wins | Median ΔLOC | Max ΔLOC | Exn.+Str.+API Share | Median \|fix - trigger\| | \|d\| <= 1 | \|d\| <= 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reflexion -> ITSSM | 16 | 8/16 (50.0%) | 0.5 | 5 | 68.9% | 1.0 | 56.3% | 75.0% |
| One-shot -> ITSSM | 19 | 10/19 (52.6%) | 0.0 | 5 | 61.0% | 12.0 | 10.5% | 31.6% |
| ITSSM-SFT -> ITSSM | 4 | 0/4 (0.0%) | 11.0 | 55 | 38.2% | 7.5 | 25.0% | 50.0% |

### 1.1 What the SFT Ablation Adds

The SFT-only prover already recovers most of the BigCodeBench gain, so the RL-trained critic should be described as a **small but real** improvement on this split, not a large jump:

- ITSSM-SFT solves `62/148`; ITSSM solves `63/148`.
- RL adds `4` wins over ITSSM-SFT and loses `3`, for a net `+1`.
- The `4` RL-only wins are `BigCodeBench/123`, `BigCodeBench/526`, `BigCodeBench/845`, and `BigCodeBench/865`.
- The `3` regressions against SFT are `BigCodeBench/341`, `BigCodeBench/461`, and `BigCodeBench/897`.

Failure-mode conversion shows where the extra gain comes from:

- SFT crash failures -> ITSSM pass: `3/23 = 13.0%`
- SFT assertion-only failures -> ITSSM pass: `1/51 = 2.0%`
- SFT mixed failures -> ITSSM pass: `0/11 = 0.0%`

So the RL-trained critic helps primarily on a small number of **crash / contract / boundary-condition** failures, not by broadly lifting all remaining SFT misses.

The Amplify root-trigger annotations tell the same story. On the `4` SFT-fail -> ITSSM-pass tasks, the final `|fix - root_trigger|` distances are `2`, `21`, `0`, and `13`, giving:

- median `7.5`
- `|d| <= 1`: `25.0%`
- `|d| <= 5`: `50.0%`

This is still localized, but noticeably less hyper-local than Reflexion -> ITSSM. The right interpretation is that RL post-training matters mainly by improving the critic's handling of a few tougher failure cases, rather than by creating a large new pool of easy same-line fixes.

## 2) The Key Reviewer Question: Are the Fixes “Surgical” or “Rewrite-y”?

We answer this using the ITSSM trace, measuring how much code changes from the pipeline’s selected initial candidate to the final submitted code:

- `LOC(initial)` = non-empty lines in `initial_proof_term`
- `changed_lines(initial→revised)` = line-diff count on non-empty lines
- `changed_ratio = changed_lines / LOC(initial)`

### 2.1 When the pipeline sees a concrete public-test counterexample
Group = tasks whose trace contains `Public test signal: FAIL` (n=114/148).

From `reports/bcb_localization_overview.json`:
- `changed_lines`: **median 1**, p90 **7**
- `changed_ratio`: **median 0.0278**, p90 **0.2268**

Interpretation: even when there is an explicit failing-test counterexample to respond to, ITSSM typically performs **1-line fixes** and stays within a small fraction of the original solution’s size (90% within ~23%).

### 2.2 Among tasks that end up passing
Group = tasks with final status `pass` (n=63/148).

From `reports/bcb_localization_overview.json` and `reports/bcb_localization_per_task.csv`:
- `changed_lines`: **median 0**, p90 **3.8**
- `changed_ratio`: **median 0**, p90 **0.1462**
- `changed_lines == 0`: **43/63 = 68.3%** of passing tasks

Interpretation: ITSSM often solves tasks by selecting a correct initial candidate, and when revisions are needed they are very small (90% within ~4 changed non-empty lines).

### 2.3 On the tasks ITSSM uniquely solves (baseline fail → ITSSM pass)
These are the tasks reviewers care about: “where did the gain come from?”.

From `reports/bcb_localization_per_task.csv`:
- Reflexion → ITSSM improved tasks (n=16): `changed_lines` median `0.5`, max `5`
  - Half of these improvements (8/16) have `changed_lines == 0`, meaning the gain is achieved by ITSSM’s **candidate search/selection** rather than a large revision.
  - For the remaining half, the patch is still small: nonzero median `2`, max `5`.
- Simple → ITSSM improved tasks (n=19): `changed_lines` median `0`, max `5`
  - `changed_lines == 0` for 10/19.
  - For the rest: nonzero median `3`, max `5`.

This is the “surgical fix” signature: when the approach gains new solved tasks, it does so by selecting a better near-miss and then applying a tiny patch (≤5 non-empty-line changes in this run’s trace).

### 2.4 Baseline comparison: final output size and “robustness bloat”
Baselines do not expose `initial→revised` traces in this repo, so we compare *rewrite-y* concerns using the **final submitted code** in each method’s `samples.jsonl`:

- Final-code length (non-empty LOC), all tasks (n=148):
  - ITSSM: median **31**, p90 **48**
  - Reflexion: median **28.5**, p90 **46.3**
  - Simple: median **44**, p90 **67.3**
- Final-code length, PASS tasks only:
  - ITSSM: median **29**, p90 **50.4**
  - Reflexion: median **28**, p90 **45.1**
  - Simple: median **46**, p90 **62**

So ITSSM’s higher pass@1 is **not** explained by outputting much longer “kitchen sink” solutions: its code length distribution is close to Reflexion and far smaller than Simple (which is both longer and less accurate).

To check for “robustness hacks”, we also measure the fraction of tasks whose final code contains `try:`, `except Exception:`, or bare `except:`:

- All tasks:
  - ITSSM: `try` **33.8%**, `except Exception` **10.1%**, bare `except` **2.7%**
  - Reflexion: `try` **30.4%**, `except Exception` **10.8%**, bare `except` **3.4%**
  - Simple: `try` **29.1%**, `except Exception` **5.4%**, bare `except` **2.7%**
- PASS tasks only:
  - ITSSM: `try` **28.6%**, `except Exception` **7.9%**, bare `except` **1.6%**
  - Reflexion: `try` **18.5%**, `except Exception` **7.4%**, bare `except` **0.0%**
  - Simple: `try` **19.6%**, `except Exception` **3.9%**, bare `except` **0.0%**

These comparisons support a clean reviewer claim: ITSSM’s gains come from **better search + targeted contract fixes**, not from large rewrites or broadly swallowing errors.
Source: `reports/bcb_localization_overview.json` key `final_solution_stats`.

### 2.5 Trigger vs. Fix Line Distance (Causal Locality)
Reviewer concern: “are improvements coming from edits concentrated near the causal fault, or from dispersed file-wide rewrites?”

For each task where a baseline **fails** but ITSSM **passes**, we compute:
- **trigger line** = baseline candidate line tied to the first failing assertion/exception (prefer traceback frame inside candidate; fallback to token-based heuristic)
- **fix line** = center line of the primary modified diff hunk in ITSSM’s passing code
- **distance** = `d_line = |fix_line - trigger_line|`

From `reports/bcb_localization_overview.json` key `trigger_fix_distance` (baseline-fail subset):
- Reflexion → ITSSM (n=16): trigger+fix identified in **10/16 (62.5%)**
  - `d_line` median **7.5**, p90 **30.4**
- Simple → ITSSM (n=19): trigger+fix identified in **9/19 (47.4%)**
  - `d_line` median **18**, p90 **64.8**
- ITSSM-Robust → ITSSM (n=17): trigger+fix identified in **10/17 (58.8%)**
  - `d_line` median **9**, p90 **26.1**

Interpretation:
- Where traceback provides a candidate-local fault frame, ITSSM’s primary edit hunk is usually near that fault locus.
- Coverage is below 100% because some failures are assertion-only without candidate-local frames, requiring noisier heuristic trigger localization.

### 2.6 Amplify GPT-5.2 Root-Trigger Analysis (User Run)
To strengthen causal localization, we also ran an Amplify `gpt-5.2` pass that infers **root trigger line** and **root trigger reason** from baseline code + traceback + revised code.

Source artifacts:
- `reports/bcb_trigger_fix_amplify_annotations.csv`
- `reports/bcb_trigger_fix_amplify_summary.json`
- methodology/readout: `README_BigCodeBench_TRIGGER_FIX_AMPLIFY.md`

Key run results (52 improved tasks total):
- Root trigger coverage: **52/52 (100%)**
- Fix line coverage: **50/52 (96.2%)**
- Final absolute distance (`|fix - root_trigger|`):
  - Reflexion -> ITSSM: median **1.0**, p90 **18.0** (n=16)
  - Simple -> ITSSM: median **12.0**, p90 **52.2** (n=19)
  - ITSSM-Robust -> ITSSM: median **3.0**, p90 **15.4** (n=17)
  - All baselines: median **4.0**, p90 **33.3** (n=52)

Pattern distribution over LLM-labeled revisions (aggregated):
- `exception_contract` **23.7%**
- `boundary_case` **18.3%**
- `api_surface` **14.5%**
- `algorithm_logic` **9.9%**

These Amplify root-cause labels reinforce the same claim as Section 2.5: ITSSM gains are predominantly targeted contract fixes, with strongest locality against stronger baselines (Reflexion and ITSSM-Robust).

### 2.6.1 Reflexion vs ITSSM: Trigger-Causal Difference
This subsection isolates the direct comparison users/reviewers usually ask for: **on tasks Reflexion fails, why does ITSSM pass?**

Population:
- Reflexion fail -> ITSSM pass: **16 tasks** (from `comparisons.improved_over_reflexion`)
- Net performance gain vs Reflexion on full benchmark: **+9 solved tasks** (63 vs 54 pass)

Trigger/fix locality on this 16-task set (Amplify `gpt-5.2` root-cause extraction):
- Root trigger coverage: **16/16 (100%)**
- Fix line coverage: **16/16 (100%)**
- `|fix - root_trigger|`: median **1.0**, mean **5.19**, p90 **18.0**, max **23.0**
- Distance bands:
  - exact same line (`d=0`): **5/16 (31.3%)**
  - within 1 line (`d<=1`): **9/16 (56.3%)**
  - within 5 lines (`d<=5`): **12/16 (75.0%)**
  - within 10 lines (`d<=10`): **13/16 (81.3%)**

Cross-baseline context for these trigger distances (all from `reports/bcb_trigger_fix_amplify_annotations.csv`):
- Reflexion -> ITSSM (`n=16`): median **1.0**, p90 **18.0**, `d<=1` **56.3%**, `d<=5` **75.0%**
- ITSSM-Robust -> ITSSM (`n=17`): median **3.0**, p90 **15.4**, `d<=1` **23.5%**, `d<=5` **64.7%**
- Simple -> ITSSM (`n=19`): median **12.0**, p90 **52.2**, `d<=1` **10.5%**, `d<=5` **31.6%**

This comparison strengthens the causal claim: the Reflexion gap is closed mostly by **local contract repairs**, while Simple often needs broader rewrites before contract-level alignment.

Compared with the earlier heuristic-only trigger estimate on the same set:
- trigger+fix measurable coverage: **62.5%** (10/16)
- median distance: **7.5** (vs **1.0** with root-cause extraction)
- p90 distance: **30.4** (vs **18.0**)

Revision-pattern differences explaining why ITSSM wins over Reflexion on these tasks:
- Dominant pattern mass: `exception_contract` (**30.8%**) + `api_surface` (**20.5%**) + `boundary_case` (**12.8%**)
- Primary pattern labels are led by `api_surface` (**6/16**) and `exception_contract` (**3/16**)
- Contract-heavy share (`exception_contract + api_surface`) on improved sets:
  - Reflexion -> ITSSM: **51.3%**
  - ITSSM-Robust -> ITSSM: **41.9%**
  - Simple -> ITSSM: **24.5%**

Interpretation:
- Reflexion failures are concentrated in thin contract bugs (mock/API surface mismatches, expected exception behavior, and boundary handling).
- ITSSM’s successful edits are usually at or near the causal trigger line, indicating **surgical, causally targeted repair** rather than diffuse rewriting.

### 2.6.2 Deeper Trigger-Line + Fix-Pattern Evidence (Reflexion-Focused)
To make the “surgical vs rewrite-y” question explicit, we compute additional derived metrics from:
- `reports/bcb_trigger_fix_amplify_annotations.csv`
- `reports/bcb_localization_overview.json`
- `reports/bcb_trigger_fix_reflexion_deep_dive.json`

Derived metrics:
- `normalized_trigger_distance = |fix - root_trigger| / LOC(revised)`
- `strict_surgical = (|fix-root_trigger| <= 5) and (hunk_changed_lines <= 10)`
- `micro_surgical = (|fix-root_trigger| <= 1) and (hunk_changed_lines <= 8)`

Cross-baseline comparison on baseline-fail -> ITSSM-pass subsets:
- Reflexion -> ITSSM (`n=16`):
  - normalized distance median: **2.67%**
  - long-tail (`d>20`): **2/16 (12.5%)**
  - median primary-hunk size: **5.5 lines**
  - strict surgical: **11/16 (68.8%)**
  - micro surgical: **9/16 (56.3%)**
- ITSSM-Robust -> ITSSM (`n=17`):
  - normalized distance median: **5.56%**
  - long-tail (`d>20`): **2/17 (11.8%)**
  - median primary-hunk size: **8.0 lines**
  - strict surgical: **8/17 (47.1%)**
  - micro surgical: **4/17 (23.5%)**
- Simple -> ITSSM (`n=19`):
  - normalized distance median: **27.94%**
  - long-tail (`d>20`): **7/19 (36.8%)**
  - median primary-hunk size: **18.0 lines**
  - strict surgical: **4/19 (21.1%)**
  - micro surgical: **2/19 (10.5%)**

Reflexion-specific pattern-locality coupling:
- Contract-like primary patterns (`exception_contract`, `api_surface`, `string_contract`, `input_validation`, `boundary_case`) cover **13/16 (81.3%)** tasks.
- On these 13 tasks:
  - median distance: **1.0**
  - p90 distance: **11.4**
  - `d<=1`: **8/13 (61.5%)**
  - `d<=5`: **11/13 (84.6%)**
- Non-contract primary patterns are only **3/16**, with weaker locality (median **8.0**, p90 **20.0**).

Failure-signal alignment inside Reflexion -> ITSSM wins:
- Baseline failing signal split:
  - exception/crash failures: **9/16**
  - assertion-only failures: **7/16**
- Primary fix patterns align with failure type:
  - crash subset: mostly `api_surface`/`type_conversion` (**7/9**)
  - assertion subset: led by `exception_contract` (**3/7**)
- Locality remains centered on the causal locus in both groups:
  - crash subset median distance: **1.0**
  - assertion subset median distance: **3.0**

Directionality and code-size sanity checks:
- Signed trigger-to-fix distance for Reflexion set is centered near zero (median **0**, with **6** above / **5** same-line / **5** below), indicating edits are not systematically drifting to unrelated file regions.
- Final solution size stays comparable to Reflexion on this subset (`loc_delta_ours_minus_baseline`: median **+1** line, mean **-0.625**), so gains are not explained by simple code inflation.

Outlier transparency:
- Only **3/16** Reflexion-win tasks have `d>10`:
  - `BigCodeBench/418` (`api_surface`, `d=23`)
  - `BigCodeBench/760` (`type_conversion`, `d=23`)
  - `BigCodeBench/574` (`api_surface`, `d=13`)
- These are the minority tail; the median behavior is still near-trigger contract repair.

## 3) Where Do These Fixes Land? (Edit-Type Localization Matches Unit-Test Contracts)

We classify changed lines (both deleted-from-initial and new/modified-in-revised) into:
- `exception_handling`: line begins with `try`, `except`, `raise`, `finally`
- `string_literal`: contains a string literal
- `api_surface`: contains `.<attr_or_method>`
- `other`

This is a deliberately simple, reviewer-friendly analysis: it tests whether improvements cluster in the code regions that unit tests typically enforce via hidden contracts.

### 3.1 Reflexion → ITSSM improvements
From `reports/bcb_localization_edit_types_reflexion_improved.json` (n=16 tasks; 29 classified changed lines total):
- `exception_handling`: **24.1%**
- `string_literal`: **24.1%**
- `api_surface`: **20.7%**
- `other`: **31.0%**

So **68.9%** of the changed lines on ITSSM’s “wins” fall into the categories most associated with hidden test contracts:
- **exception contracts** (exact exception type must be raised)
- **format/string contracts** (exact phrase required)
- **API surface correctness** (correct attribute/method for mocked objects)

This matches the observed failure modes in the Reflexion evaluator logs (examples are enumerated in `README_BigCodeBench_MOTIVATION_EXAMPLE_SUBGOAL_FIX.md`).

### 3.2 Simple → ITSSM improvements
From `reports/bcb_localization_edit_types_simple_improved.json` (n=19; 36 classified changed lines):
- `exception_handling`: **19.4%**
- `string_literal`: **22.2%**
- `api_surface`: **19.4%**
- `other`: **38.9%**

Even against a one-shot baseline, **61.0%** of the change mass still concentrates on exception/string/API-surface edits, i.e. the exact places where “nearly correct” solutions tend to fail unit tests.

## 4) Diagnostic: ITSSM Reduces Runtime “Crash” Failures (Not Just Assertion Tweaks)

A common concern is whether improvements come from fragile string hacks. We therefore also look at a coarse failure-mode breakdown from `*_eval_results.json`:
- `Fail-Crash` = at least one failing test ends in a non-`AssertionError` exception (e.g., `TypeError`, `AttributeError`)
- `Fail-AssertionOnly` = all failing tests are `AssertionError`
- `Fail-Mixed` = mixture

Counts (148 tasks):
- **ITSSM:** `Pass 63`, `Fail-Crash 21`, `Fail-AssertionOnly 52`, `Fail-Mixed 11`, `Timeout 1`
- **Reflexion:** `Pass 54`, `Fail-Crash 33`, `Fail-AssertionOnly 50`, `Fail-Mixed 10`, `Timeout 1`
- **Simple:** `Pass 51`, `Fail-Crash 30`, `Fail-AssertionOnly 53`, `Fail-Mixed 14`

Interpretation:
- ITSSM reduces **crash failures** substantially compared to both baselines:
  - vs Reflexion: `33 → 21` (−12 tasks, ~36% reduction)
  - vs Simple: `30 → 21` (−9 tasks, ~30% reduction)
- This is consistent with the edit-type evidence: many wins are **API-surface** or **exception-handling** fixes that prevent runtime failures under mocks and edge cases.

This failure-mode shift supports a stronger claim than “we changed strings”: ITSSM makes code *execute correctly under the harness* more often, then satisfies the remaining assertion-level contracts.

## 5) Why This Pattern Is Expected (Mechanism → Observed Metrics)

BigCodeBench (and similar unit-test-driven code benchmarks) reward **spec adherence**, where many errors are “thin”:
- wrong exception type (e.g., `JSONDecodeError not raised`)
- missing required phrase in output
- using the wrong mocked attribute/method (`.text` vs `.content`, `.raise_for_status()` semantics, etc.)

ITSSM is designed to fix precisely these thin, test-enforced contracts:
- The prover-critic emits **explicit subgoals** (obligations) instead of free-form self-critique.
- The pipeline can inject **concrete counterexamples** (failing test signal) during revision and selection.
- Low revision temperature and multi-sample selection bias revisions toward **minimal diffs** (reduce collateral damage).

The measured evidence matches these expectations:
- Most final-pass tasks require no changes (search/selection finds a correct candidate).
- When changes are needed, they are small (median 1 line on public-fail tasks).
- The changes concentrate where contracts live: exception/string/API-surface lines (≈61–69% of “win” edits).

## 6) Why This Should Transfer Beyond BigCodeBench

The fixation pattern above is not BigCodeBench-specific; it is a property of test-driven evaluation:
- Unit tests enforce a mixture of semantic and syntactic contracts (exception classes, substrings, API surfaces under mocks).
- These contracts are typically satisfiable by **localized patches** once the correct obligation is identified.

Evidence that the same ITSSM pipeline is already applied beyond BigCodeBench:
- LiveCodeBench `v6` best run documented in `README_ITSSM_BEST_RUN_1771167978.md`:
  - `pass@1 = 0.3943` (69/175) using the same Stage 1/2/3 design (coder + RL critic + search) and the same “public self-test as selection signal” pattern.

Reviewer-facing framing:
- ITSSM can be presented as “a learned debugger” that converts latent unit-test contracts into explicit obligations and then applies a minimal patch.
- Since other code benchmarks (LiveCodeBench, HumanEvalFix-like repair suites, SWE-style unit tests) share the same unit-test contract structure, the same localization signature is expected to recur.

## 7) Reproduce the Numbers in This Report

```bash
cd /home/yueke/formalgen/lean_gen
python scripts/bcb_localization_report.py
```

This regenerates the `reports/bcb_localization_*.{json,csv}` files referenced above.
