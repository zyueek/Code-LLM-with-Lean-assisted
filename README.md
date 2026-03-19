# ITSSM Artifact Bundle

This directory is a standalone bundle of the files behind the two referenced best-run documents:

- `lean_gen/README_ITSSM_BEST_RUN_1771167978.md`
- `lean_gen/README_BigCodeBench_BEST_RUN_1771347697.md`

The layout intentionally keeps a partial `lean_gen/` tree so the copied scripts still match the paths used in the original repo docs. `vllm_client.py` is placed at the bundle root because the main ITSSM scripts expect it one directory above `lean_gen/`.

## What is included

- `lean_gen/README_ITSSM_BEST_RUN_1771167978.md`
  - Methodology and exact command for the best LiveCodeBench v6 ITSSM run (`1771167978`).
- `lean_gen/README_BigCodeBench_BEST_RUN_1771347697.md`
  - Methodology and exact command for the best BigCodeBench-Hard ITSSM run (`1771347697`).
- `lean_gen/README_BigCodeBench_ITSSM_BETTER_THAN_BASELINES.md`
  - Comparison note against Reflexion, one-shot Qwen3, and the SFT-only ablation.
- `lean_gen/README_BigCodeBench.md`
  - Baseline evaluation guide for BigCodeBench.

## Script guide

- `lean_gen/fdg_approach_qwen3_inter_posttrained.py`
  - Main LiveCodeBench ITSSM inference runner for the best run. It uses Qwen3 as the coder and a post-trained DeepSeek Prover critic loaded either from vLLM or a local HuggingFace/PEFT checkpoint.
- `lean_gen/fdg_approach_qwen3_posttrained_bigcodebench.py`
  - BigCodeBench version of the same post-trained ITSSM pipeline. It adds BigCodeBench task formatting and writes `samples.jsonl` for official evaluation.
- `lean_gen/fdg_approach_qwen3_inter_refined_reflex_robust.py`
  - Pre-posttraining ITSSM robust-reflex baseline. This is the base pipeline that the post-trained variant extends.
- `lean_gen/generate_itssm_rl_dataset_livecodebench.py`
  - Replays the ITSSM pipeline on LiveCodeBench to create JSONL training targets for the critic.
- `lean_gen/rl_training_prover_sft.py`
  - Supervised fine-tuning warm-start for the critic. It teaches the prover to emit the exact Stage-2 tag format used at inference time.
- `lean_gen/rl_training_prover_process.py`
  - GRPO/RL training entry point for the critic. Rewards are based on how much the coder improves after conditioning on the critic output.
- `lean_gen/configs/rl_prover_grpo_v3_mix_from_v2_gpt52.args`
  - Canonical config for the v3 critic used by the best runs.
- `lean_gen/reflexion_feedback_qwen3_bigcodebench.py`
  - BigCodeBench Reflexion baseline used for comparison.
- `lean_gen/simple_qwen3_bigcodebench.py`
  - BigCodeBench one-shot Qwen3 baseline used for comparison.
- `lean_gen/reflexion_feedback_qwen3_improved.py`
  - Supporting Reflexion implementation imported by the BigCodeBench Reflexion wrapper.
- `lean_gen/simple_qwen3_livebench.py`
  - Supporting simple Qwen3 generator imported by the BigCodeBench one-shot wrapper.
- `lean_gen/train_deepseek_prover_rl.py`
  - Shared utilities used by dataset generation and RL training, including LiveCodeBench loading and executor helpers.
- `vllm_client.py`
  - Shared OpenAI-compatible vLLM client dependency used by the ITSSM scripts.

## Artifact guide

- `lean_gen/itssm_approach_results/`
  - Saved LiveCodeBench best-run trace and evaluation outputs for run `1771167978`.
- `lean_gen/itssm_approach_results_sft_ablation/`
  - Saved LiveCodeBench SFT-only ablation evaluation output cited in the README.
- `lean_gen/bcb_results/`
  - BigCodeBench best-run sample/evaluation files and the Reflexion baseline sample/evaluation files.
- `lean_gen/simple_results/`
  - BigCodeBench one-shot Qwen3 baseline sample/evaluation files.
- `lean_gen/bcb_results_sft_ablation/`
  - BigCodeBench SFT-only critic ablation sample/evaluation files.

## Not bundled

This bundle does not include:

- model weights or PEFT adapter directories such as `rl_prover_critic_v3`
- the full `bigcodebench/` repo checkout
- cached benchmark datasets under `~/.cache`
- API keys or external service credentials

Use the two best-run READMEs first if you need the exact reproduction commands.
