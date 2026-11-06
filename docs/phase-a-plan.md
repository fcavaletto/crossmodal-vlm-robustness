# Phase A plan — repository validation and clean baselines

**Repo:** `https://github.com/fcavaletto/crossmodal-vlm-robustness`  
**Checkout:** `/Users/fedecava/Development/crossmodal-vln-robustness` (empty worker label dir; cloned once into it)  
**Hardware:** MacBook Air M4, 24 GB unified memory, MPS only  
**Scope:** Phase A only — no attacks, no held-out test evaluation, no Phase B gradients

---

## 1. Inspection summary

### What exists and works (code/data, not yet model-verified)

| Component | Status |
| --- | --- |
| Deterministic Pillow generator (`generate_dataset.py`, seed `20261105`) | Implemented; 96 images checked in |
| Balanced manifest + SHA-256 hashes + metadata + preview contact sheet | Implemented under `data/synthetic_v1/` |
| Prompt builder: neutral / supportive / misleading with fixed question wording | Implemented (`dataset.build_prompt`) |
| Strict answer parser (reject verbose/ambiguous) | Implemented (`dataset.parse_answer`) |
| Baseline evaluator CLI (`evaluate.py`): dry-run, MPS/CPU, JSONL logging, summary | Implemented; **never run with real weights in this workspace** |
| Generic `pgd_linf` primitive + toy unit test | Implemented; **not** wired to VLM loss (Phase B/C) |
| Unit tests for dataset balance, prompts, parser, PGD bounds | Present |

### What is planned / not in Phase A

- Differentiable VLM image→loss adapter and nonzero gradient check (Phase B)
- PGD/FGSM/noise attacks, paired evaluation, bootstrap CIs, plots (Phases C–D)
- Held-out test evaluation and technical report freeze (Phases D–E)
- Optional Qwen3-VL cross-model validation

### Bugs / gaps / risks spotted before coding

1. **Model API unverified.** `AutoModelForImageTextToText` + `processor.apply_chat_template` with SmolVLM2 may need processor/message-format fixes on current Transformers.
2. **MPS / float32 / eager attention** may be slow or OOM on 24 GB if video-capable model allocates large intermediates; smoke test first.
3. **Strict parser** may yield high invalid rates if SmolVLM2 answers verbosely — scientifically honest but may tank “accuracy”; report invalid rate separately.
4. **Python on machine:** Homebrew has 3.12 and 3.14; prefer **3.12** for PyTorch wheel compatibility.
5. **No SmolVLM2 in local HF cache** yet — first run downloads weights (~500M + processor).
6. **Scientific risk:** synthetic stimuli may be trivial or oddly hard for a VLM; Phase A baselines tell us which. Shared images across color/shape tasks → non-independent observations (protocol already notes this).
7. **Worker path naming** (`crossmodal-vln-robustness` vs `vlm`) is cosmetic only; single clone used.

### Preferred stack (no swaps)

PyTorch MPS · Transformers · NumPy · Pillow · Matplotlib · pytest · frozen `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` · float32 · eager attention

---

## 2. Phase A execution plan

1. **Branch:** `cursor/phase-a-baseline-dabf` from `main`.
2. **Env:** `python3.12 -m venv .venv`; `pip install -e '.[dev,plot]'`.
3. **Tests:** `pytest -q`; dry-run evaluator; fix only Phase-A compatibility issues.
4. **Dataset integrity:** verify 96 PNGs, hashes, 64/32 splits, 4×4×6 balance, wrong labels ≠ true.
5. **Hardware smoke:** confirm `torch.backends.mps.is_available()`; load SmolVLM2; infer ≥2 contrasting images; log peak memory + latency + package/model revisions.
6. **Vision check:** manually inspect contrasting color/shape pairs under neutral prompts; confirm different images → different (or at least image-sensitive) outputs; verify pairing/parser.
7. **Development baselines:** full development split (`--max-examples 0`), tasks both, conditions Neutral/Supportive/Misleading, `do_sample=False`. **Do not touch test split.**
8. **Deliverables:** `docs/phase-a-verification.md` (store + repo mirror), `internal/phase-a-status.md`; commit + push feature branch; **stop for review**.

### Explicit non-goals this phase

No PGD against the VLM · no gradient debugging · no test-set inference · no main merge · no MLX/GGUF/API substitutes
