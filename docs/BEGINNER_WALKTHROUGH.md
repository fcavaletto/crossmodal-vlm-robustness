# Beginner walkthrough (rigorous but accessible)

This walkthrough teaches the **estimands** used in the study. It does not replace
[`reports/technical_report.md`](../reports/technical_report.md). No model run is required.

---

## 1. What is being asked?

We have cartoon images with a known **color** and **shape**. The model also reads a
short **context** sentence that is one of:

| Context | Content |
| --- | --- |
| Neutral | No color/shape claim |
| Supportive | States the **true** label |
| Misleading | States a **false** label |

Then it answers a one-word color or shape question. A **strict parser** marks the
answer correct, incorrect, or invalid (invalid counts as error).

Separately, we may change the **pixels** inside an L∞ ball of radius ε (FGSM, PGD,
or random noise). The attack is built with the **neutral** prompt. The **same**
perturbed image is then scored under all three contexts. That pairing asks:

> Does misleading text make the attack *add more errors* than supportive text?

---

## 2. Error, additional error, and DiD

For one image and one context \(c\):

\[
e^{\mathrm{clean}}_c,\; e^{\mathrm{adv}}_c \in \{0,1\}
\]

(1 = wrong or invalid).

**Additional error**

\[
\Delta_c = e^{\mathrm{adv}}_c - e^{\mathrm{clean}}_c \in \{-1,0,1\}.
\]

**Primary DiD** (study primary estimand)

\[
\mathrm{DiD}_{\mathrm{mis-sup}} = \mathrm{mean}_{\text{images}}\big(\Delta_{\mathrm{misleading}} - \Delta_{\mathrm{supportive}}\big).
\]

If misleading *amplified* attacks relative to supportive, we would expect this mean
**> 0**. On held-out **color**, the published mean is **negative** (e.g. −0.500 at
ε=8/255 for PGD). That is evidence **against** amplification on this benchmark.

**Secondary DiD** (not a defense claim)

\[
\mathrm{DiD}_{\mathrm{neu-sup}} = \mathrm{mean}_{\text{images}}\big(\Delta_{\mathrm{neutral}} - \Delta_{\mathrm{supportive}}\big).
\]

Positive values mean neutral suffers *more additional* attack error than supportive.

---

## 3. Worked example A — one held-out image

From the committed compact table
`reports/artifacts/phase_e_heldout_predictions_compact.jsonl`
(derived from the held-out run; **no new inference**):

**Image** `item_004`, **task** color, **attack** PGD, **ε** = 8/255.

| Context | Clean parsed | Adv parsed | \(e^{\mathrm{clean}}\) | \(e^{\mathrm{adv}}\) | \(\Delta\) |
| --- | --- | --- | --- | --- | --- |
| neutral | red | blue | 0 | 1 | +1 |
| supportive | red | red | 0 | 0 | 0 |
| misleading | green | green | 1 | 1 | 0 |

- Contribution to **primary** DiD: \(\Delta_{\mathrm{mis}}-\Delta_{\mathrm{sup}}=0-0=0\).  
- Contribution to **secondary** DiD: \(\Delta_{\mathrm{neu}}-\Delta_{\mathrm{sup}}=1-0=1\).

**Reading:** on this image, the attack flips the neutral answer but not the supportive
one; misleading was already wrong before the attack (floor), so the attack cannot
“add” more misleading error.

---

## 4. Worked example B — why the floor matters

Suppose 28 of 32 clean misleading-color answers are already wrong. Then for those 28
images, \(\Delta_{\mathrm{mis}}\) cannot be \(+1\) (error is already 1). The primary DiD
compares *additional* errors. Low clean accuracy under misleading is therefore not
evidence that attacks were amplified; it is a **ceiling on identifiable additional
error**. Held-out misleading-color clean accuracy is **4/32 = 12.5%** (committed metrics).

---

## 5. Bootstrap in one paragraph

After computing one DiD contribution per image, we resample the **32** (or 64)
image contributions with replacement, 2000 times, seed `20261105`, and take the
2.5% and 97.5% percentiles. That is the published CI. Implementation:
`crossmodal_robustness.phase_d_analyze.bootstrap_did`. You can re-check without a
GPU via:

```bash
python -m crossmodal_robustness.verify_published
```

---

## 6. Attack objective (why “mean NLL”?)

Attacks ascend the **mean** teacher-forced NLL of the correct answer tokens (HuggingFace
`loss` with prompt positions masked). See technical report §6.2. Evaluation of success
still uses **greedy generation + strict parser**, not the NLL alone.

---

## 7. What this study does *not* claim

- That supportive text is a general defense.  
- Mechanistic explanations (no attention / probing analysis).  
- Real-world safety rates.  
- That larger ε always yields stronger greedy ASR (some cells are non-monotone).

## Next reading

1. README “shows vs does not show”  
2. Technical report §§3, 6–9  
3. `docs/REPRODUCIBILITY.md`  
4. `reports/phase-e-verification.md`
