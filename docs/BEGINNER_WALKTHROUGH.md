# How to read this study (no GPU required)

This note explains the *ideas* behind the numbers. It does not replace
[`reports/technical_report.md`](../reports/technical_report.md), and it does not
run the model.

## The setup

Imagine a cartoon: a large red circle on a blank background. You know the color
and shape because a script drew them. The model sees that image plus a short
**context** sentence, then answers a one-word question (“What COLOR…?” or
“What SHAPE…?”).

The context is one of three kinds:

| Context | What the sentence does |
| --- | --- |
| Neutral | Describes “one large central figure” without naming color or shape |
| Supportive | States the **true** label |
| Misleading | States a **false** label |

A **strict parser** accepts only an exact allowed word. Anything verbose,
ambiguous, or off-list counts as an **error** (same as a wrong label).

Now change the *pixels* a little—inside an L∞ ball of radius ε—using FGSM, PGD,
or equal-budget random noise. We build that perturbation with the **neutral**
prompt, then lock the pixels and ask again under all three contexts.

The scientific question is not “is the attack strong?” in isolation. It is:

> Relative to a truthful sentence, does a *false* sentence make the same attack
> *add more mistakes*?

## From one image to a DiD

For one image and one context \(c\), write \(e^{\mathrm{clean}}_c\) and
\(e^{\mathrm{adv}}_c\) for whether the answer is wrong (1) or right (0). The
**additional error** is

\[
\Delta_c = e^{\mathrm{adv}}_c - e^{\mathrm{clean}}_c \in \{-1,0,1\}.
\]

The study’s **primary** number averages, over images,

\[
\mathrm{DiD}_{\mathrm{mis-sup}} = \mathrm{mean}\big(\Delta_{\mathrm{misleading}} - \Delta_{\mathrm{supportive}}\big).
\]

If misleading *amplified* the attack relative to supportive, this mean would be
positive. On held-out **color**, the published mean is **negative** (for PGD at
ε=8/255 it is **−0.500**). That is evidence **against** amplification on this
benchmark.

A **secondary** average replaces misleading with neutral:

\[
\mathrm{DiD}_{\mathrm{neu-sup}} = \mathrm{mean}\big(\Delta_{\mathrm{neutral}} - \Delta_{\mathrm{supportive}}\big).
\]

Positive values mean the attack added more errors under neutral than under
supportive. Useful as a description of context sensitivity—not as a claim that
supportive text is a general defense.

## Worked example: one held-out image

The row below comes from the committed compact table
`reports/artifacts/phase_e_heldout_predictions_compact.jsonl` (derived from the
held-out run; no new inference). Image `item_004`, color task, PGD at ε=8/255:

| Context | Clean answer | After PGD | \(e^{\mathrm{clean}}\) | \(e^{\mathrm{adv}}\) | \(\Delta\) |
| --- | --- | --- | --- | --- | --- |
| neutral | red | blue | 0 | 1 | +1 |
| supportive | red | red | 0 | 0 | 0 |
| misleading | green | green | 1 | 1 | 0 |

This image contributes \(0\) to the primary DiD and \(1\) to the secondary DiD.
The attack flipped the neutral answer; supportive stayed correct; misleading was
already wrong before any perturbation, so the attack could not “add” a
misleading error.

## Why a clean “floor” blocks the amplification story

On held-out data, misleading-color clean accuracy is only **4/32 = 12.5%**. For
images that are already wrong under misleading, \(\Delta_{\mathrm{mis}}\) cannot
be \(+1\). The primary DiD compares *additional* errors. A low clean misleading
accuracy is therefore not proof that attacks got worse under conflict—it is a
**ceiling on how much additional misleading error you can even see**.

In fact, **28 of the 32** held-out color questions were already answered incorrectly under misleading text, so at most four could acquire a *new* error from an image attack in that condition.

## Uncertainty without a GPU

After one DiD contribution per image, we resample the 32 (or 64) contributions
with replacement, 2000 times, seed `20261105`, and take the 2.5% / 97.5%
percentiles. That is the published confidence interval
(`crossmodal_robustness.phase_d_analyze.bootstrap_did`).

```bash
python -m crossmodal_robustness.verify_published
```

rebuilds those statistics from committed compact tables and checks them against
the published metrics JSON—no model download required.

## What the attack optimizes (and what we score)

Attacks ascend the **mean** teacher-forced negative log-likelihood of the correct
answer tokens (HuggingFace’s usual `loss` with prompt positions masked). That is
the training-style objective used to move pixels. Whether an attack “worked” for
the paper’s tables is still decided by **greedy generation plus the strict
parser**, not by the NLL alone. See technical report §6.2.

## What not to take away

Supportive text is not shown to be a general defense. We offer no attention maps
or probing analysis. Numbers here are not real-world safety rates. Larger ε does
not always produce higher greedy attack-success rates on this stack (some cells
are non-monotone).

## Next

README for the public summary; technical report §§3 and 6–9 for definitions and
tables; [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for commands; `reports/phase-e-verification.md`
for the held-out lab log.
