# Freeze configuration (machine-readable)

**freeze_id:** `checkpoint-1-2026-11-05`  
**status:** frozen  
**code_tip_branch:** `cursor/phase-d-dev-eval-dabf`  
**code_tip_commit:** `17d64b2`  
**protocol:** `docs/PROTOCOL.md`  
**federico_decisions:** `D1(c)`, `D2`  

## Decisions

```yaml
D1: both_dids_original_scientific_status
  primary_did: misleading_minus_supportive   # preregistered; report even if negative
  secondary_did: neutral_minus_supportive    # strongest secondary; not a validated defense
  rewrite_hypothesis_retrospectively: false
  promote_exploratory_as_preregistered: false

D2: freeze_phase_d_as_run
  epsilon_set: [2/255, 4/255, 8/255]   # full as-run grid — NOT ε=8/255 alone
  note: >
    Informal “ε ∈ 8/255” wording is treated as a typo relative to Phase D,
    which validated {2,4,8}/255. Ceiling at ε=2/255 is a reported limitation.
  retune_for_prettier_results: false
```

## Model

```yaml
model_id: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
revision: 7b375e1b73b11138ff12fe22c8f2822d8fe03467
dtype: float32
attn_implementation: eager
weights: frozen
device: mps
```

## Environment (recording host)

```yaml
python: "3.12.14"
torch: "2.15.0"
torchvision: "0.29.2"
transformers: "5.22.1"
platform: macOS-arm64-M4-24GB
```

## Data

```yaml
dataset: synthetic_v1
generator_seed: 20261105
manifest: data/synthetic_v1/manifest.jsonl
development_n: 64
test_n: 32
test_sealed_until_after_freeze: true
```

## Prompts / parser

```yaml
prompt_module: crossmodal_robustness.dataset.build_prompt
parser_module: crossmodal_robustness.dataset.parse_answer
parser: strict_exact_match
invalid_counts_as_error: true
prompts_frozen: true
```

## Preprocess / tokens

```yaml
do_image_splitting: false
attack_input_space: rgb_float_chw_[0,1]
image_processor_do_rescale_on_attack_path: false
eval_roundtrip: float_adv -> uint8_pil -> generate
answer_tokenization: leading_space_bpe  # e.g. Ġred
phase_b_bare_token_nll: superseded_erratum
```

## Attacks (as-run)

```yaml
optimize_condition: neutral
evaluate_conditions: [neutral, supportive, misleading]
reuse_same_perturbation_across_conditions: true
epsilons:
  - 0.00784313725490196   # 2/255
  - 0.01568627450980392   # 4/255
  - 0.03137254901960784   # 8/255
pgd:
  steps: 7
  step_size: epsilon_over_4
  random_start: false
fgsm: true
random_linf_noise: true
seed: 20261105
objective: teacher_forced_nll_correct_answer_ascent
```

## Generation

```yaml
do_sample: false
max_new_tokens: 12
```

## Statistics

```yaml
did_unit: image
bootstrap_resamples: 2000
bootstrap_seed: 20261105
ci: percentile_95
tasks_reported_separately: true
```

## Held-out (planned; not executed at Checkpoint 1)

```yaml
run_heldout: false_at_checkpoint_1
when_authorized: after_this_freeze_only
split: test
procedure: identical_to_phase_d_frozen_grid
post_hoc_retune_forbidden: true
```

## Entry points

```yaml
dev_eval: python -m crossmodal_robustness.phase_d_eval --subset all --no-do-image-splitting ...
analyze: python -m crossmodal_robustness.phase_d_analyze --jsonl results/phase_d_full64.jsonl
heldout_validate: python -m crossmodal_robustness.heldout_eval --validate-only
heldout_eval: python -m crossmodal_robustness.heldout_eval --i-authorize-heldout-oneshot ...  # see docs/pre-test-addendum-2026-11-05.md
heldout_analyze: python -m crossmodal_robustness.heldout_analyze --jsonl results/phase_e_heldout.jsonl --development-metrics reports/phase-d-full64-metrics.json
tests: python -m pytest -q
pre_test_addendum: docs/pre-test-addendum-2026-11-05.md
```
