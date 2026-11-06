# Beginner walkthrough

This is a conceptual path into the project. It does not replace the technical report.

## The setup in one page

You have cartoon-like images with a known **color** and **shape**. The model sees the image plus a short **context** sentence that is either:

- **Neutral** — no color/shape claim  
- **Supportive** — states the true label  
- **Misleading** — states a false label  

Then it is asked a one-word color or shape question. We score answers with a strict parser (wrong or garbled → error).

Separately, we may **perturb the pixels** inside an L∞ ball of radius ε (FGSM, PGD, or random noise). The perturbation is built using the **neutral** prompt, then the *same* perturbed image is scored under all three contexts. That pairing is what lets us ask whether misleading text makes the attack *worse* than supportive text.

## The primary number

For each image, compute how much the attack *adds* to the error rate in a context:

\[
\Delta = e_{\mathrm{after}} - e_{\mathrm{before}}
\]

Primary contrast:

\[
\mathrm{DiD} = \mathrm{mean}(\Delta_{\mathrm{misleading}} - \Delta_{\mathrm{supportive}}).
\]

If misleading *amplified* attacks, we would expect DiD > 0. On held-out color with PGD, we instead see DiD < 0 (see README / technical report). A major reason: under misleading color, the model is already wrong most of the time **before** any attack (the “floor”).

## A single-image story

Held-out `item_004`, color, PGD at ε=8/255 (from existing predictions, not a new run):

- Clean + neutral → `red` (correct)  
- PGD + neutral → `blue` (error) → Δ = 1  
- Clean/PGD + supportive → `red`/`red` → Δ = 0  
- Clean/PGD + misleading → already wrong (`green`) both times → Δ = 0  

This image contributes 0 to the primary DiD and 1 to the secondary (neutral − supportive) DiD. The study average pools many such images and bootstraps for uncertainty.

## What to read next

1. README “shows vs does not show”  
2. `reports/technical_report.md` §6–§8  
3. `docs/REPRODUCIBILITY.md` for commands and checksums  
4. `reports/phase-e-verification.md` for the lab log of the held-out run
