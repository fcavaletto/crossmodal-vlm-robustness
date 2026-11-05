# Cursor Agent Handoff — implement, verify, then measure

This repository provides the dataset, preliminary baseline evaluator, and a
generic PGD optimizer. Do not assume any model experiment has already run.
The goal is a **credible technical demonstration** before 16 Nov 2026.

## Paste this into Cursor Agent

> You are acting as a research engineer on my project `crossmodal-vlm-robustness`.
> Read `README.md`, `docs/PROTOCOL.md`, and all source/tests before changing code.
> The target machine is a **MacBook Air M4 with 24 GB unified memory**.
> Use **PyTorch on MPS** and the open-weight
> `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` model through Transformers;
> do not silently replace PyTorch with MLX, GGUF, or an API model.
>
> Deliver the implementation in **gated phases**, stopping after each phase
> with actual commands, outputs, and unresolved risks. Never invent metrics.
>
> **Phase A: validate baseline.** Install dependencies, run tests, confirm
> deterministic dataset generation, and run a two-image MPS smoke test.
> Investigate and fix model/processor API compatibility if needed. Verify that
> images genuinely reach the vision encoder; compare image inputs on a trivial
> contrasting pair. Log exact package/model revisions and memory/timing.
> Keep test-split inference off until evaluation parameters are frozen.
>
> **Phase B: gradient feasibility.** Freeze model weights; check a single
> pixel-space input tensor can receive a nonzero finite gradient from the
> correct-answer negative log-likelihood (or a justified targeted objective).
> Carefully handle image normalization, special image tokens and label token
> sequences. Any resize/crop inside the optimization must be differentiable.
> If end-to-end MPS backward is unsupported, document exactly why and propose
> a CLIP-source / VLM-transfer alternative; do not claim direct VLM PGD.
>
> **Phase C: attacks.** Connect `pgd_linf` to the differentiable VLM loss.
> Add FGSM and equal-budget random-noise controls, and unit tests verifying
> the epsilon bound in raw `[0,1]` RGB space, valid pixels, gradients and
> monotonicity on an analytic toy loss. Reuse each fixed perturbed image across
> neutral, supportive, misleading contexts. Start with development only.
>
> **Phase D: evaluate.** Implement image-paired outcomes, source success,
> baseline-correct attack success, strict parsing invalid-rate, and image-level
> bootstrap CIs for interaction difference-in-differences. Produce real plots
> and a few unmodified/perturbed image examples with pixel-difference maps.
> Freeze hyperparameters and protocol *before* the first test-split evaluation.
>
> **Phase E: report.** Update README with reproducible commands and exact
> evidence. Write a compact `reports/technical_report.md` with abstract,
> hypothesis, threats, methods, results, negative findings, limitations and
> future work. Do not claim novelty without checking related work.
>
> Rules: prioritize correctness over speed; no dataset leakage; no changing
> held-out samples; no fabricated results; log every fallback. Prefer small
> diffs and run relevant tests after changes. Ask me to approve experimental
> design choices that materially affect the scientific conclusion.

## Immediate success criteria

1. `python -m pytest -q` green.
2. `python -m crossmodal_robustness.evaluate --dry-run` shows paired prompts.
3. Real SmolVLM2 image-conditioned generations work on MPS (not assumed).
4. One nonzero finite gradient from image to candidate answer loss on MPS,
   **or** a documented CLIP transfer fallback.
5. A reproducible run script can recreate results from pinned model revision.

## Things the agent must not do

- Do not present the generic PGD primitive as a completed adversarial VLM attack.
- Do not equate random corruption with a gradient-based adversarial attack.
- Do not evaluate the held-out split while designing attacks.
- Do not report success solely on cases already incorrect before an attack.
- Do not add thousands of unlabeled Internet images without checking rights.
- Do not spend days on LoRA training—the research question does not need it.

