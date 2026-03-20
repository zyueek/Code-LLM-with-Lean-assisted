# LEAP Artifact Bundle

This directory artifact link for ASE 2026 submission "LEAP: Lean-Assisted Python Code Generation"


## Script guide

- `lean_gen/fdg_approach_qwen3_inter_posttrained.py`
  - Main LiveCodeBench SCOPE inference runner for the best run. It uses Qwen3 as the coder and a post-trained DeepSeek Prover critic loaded either from vLLM or a local HuggingFace/PEFT checkpoint.
- `lean_gen/fdg_approach_qwen3_posttrained_bigcodebench.py`
  - BigCodeBench version of the same post-trained SCOPE pipeline. It adds BigCodeBench task formatting and writes `samples.jsonl` for official evaluation.
- `lean_gen/fdg_approach_qwen3_inter_refined_reflex_robust.py`
  - Pre-posttraining SCOPE robust-reflex baseline. This is the base pipeline that the post-trained variant extends.
- `lean_gen/generate_SCOPE_rl_dataset_livecodebench.py`
  - Replays the SCOPE pipeline on LiveCodeBench to create JSONL training targets for the critic.
- `lean_gen/rl_training_prover_sft.py`
  - Supervised fine-tuning warm-start for the critic. It teaches the prover to emit the exact Stage-2 tag format used at inference time.
- `lean_gen/rl_training_prover_process.py`
  - GRPO/RL training entry point for the critic. Rewards are based on how much the coder improves after conditioning on the critic output.
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
  - Shared OpenAI-compatible vLLM client dependency used by the SCOPE scripts.

## Artifact guide

- `lean_gen/SCOPE_approach_results/`
  - Saved LiveCodeBench best-run trace and evaluation outputs for run `1771167978`.
- `lean_gen/SCOPE_approach_results_sft_ablation/`
  - Saved LiveCodeBench SFT-only ablation evaluation output cited in the README.
- `lean_gen/bcb_results/`
  - BigCodeBench best-run sample/evaluation files and the Reflexion baseline sample/evaluation files.
- `lean_gen/simple_results/`
  - BigCodeBench one-shot Qwen3 baseline sample/evaluation files.
- `lean_gen/bcb_results_sft_ablation/`
  - BigCodeBench SFT-only critic ablation sample/evaluation files.

