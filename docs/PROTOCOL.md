# Preregistered-style experimental protocol (draft)

This is an **experimental plan**, not a record of completed model evaluations.
Freeze a final version after the development pilot and before first use of
the held-out 32-image test split.

## Narrow research claim

**Question:** On simple labeled images, does contradictory textual context
increase the additional error caused by a fixed visual adversarial perturbation,
relative to a truthful textual context?

The study does not measure general-purpose safety, real-world attack rates,
or interpretability of internal mechanisms. It probes one modality interaction.

## Threat model

- Defender: frozen open-weight SmolVLM2-500M vision-language model.
- Authorized task: answer a specific question about the visual appearance of
  one central geometric figure.
- Image adversary: can modify RGB image pixels within `L_inf <= epsilon`,
  keeping valid pixel values in `[0,1]`; model parameters cannot change.
- Text adversary: supplies a false *descriptive statement* adjacent to the
  user's visual question. This is semantic misinformation, **not** an attempt
  to override the system message or a claim about instruction hierarchy.
- White-box attack: gradients are available on the development model, but
  no gradients are used during held-out evaluation beyond the specified attack.

## Stimuli and splits

- 96 deterministic 256x256 PNGs, equally represented across color and shape.
- 64 development images; 32 test images; shared image IDs across both questions.
- Questions: class of color (4 options), or shape (4 options).
- Text contexts: neutral, supportive, misleading. Supportive and misleading
  contexts use the same sentence template and differ in only the claimed label.
- Wrong-label assignment is fixed independently of model outputs.
- Examples are not natural photographs; they may be too easy for adversarial
  gradients or too synthetic for a small VLM. This limitation is explicit.

## Primary analysis

Define an error indicator `e(image, task, perturbation, context)`.
If the image attacker uses the **same** perturbation under each text context,
the difference-in-differences estimate is:

```text
interaction =
  (error_rate[PGD, misleading] - error_rate[clean, misleading])
  - (error_rate[PGD, supportive] - error_rate[clean, supportive])
```

The **supportive** context is the primary comparison because it is paired
with misleading text using the same sentence structure. Neutral context is a
secondary diagnostic. Positive interaction would indicate that misleading
context amplifies PGD-associated error *on this benchmark*, not establish
general causality in natural settings.

Use the image as the unit for paired bootstrap resampling: each image produces
two tasks and all paired conditions, so treating these observations as
independent exaggerates certainty. Report raw counts, rates, 95% bootstrap CIs,
and malformed/ambiguous response rates. Keep generated examples and data splits
fixed before selecting comparisons.

## Attack implementation — follow-up work

1. Start with a differentiable image-to-loss adapter. Test forward+backward on
   a single image using `torch.float32`, MPS, and eager attention.
2. Optimize a negative log-likelihood objective for the correct one-token
   answer **or** a carefully chosen target wrong answer. Score *full answer
   token sequences* when candidate labels tokenize differently.
3. Do not pass a PIL conversion inside the gradient path. Apply perturbations
   in `[0,1]` pixel space, then use differentiable resize/normalization as
   needed, preserving the true pixel-level epsilon bound.
4. Use fixed epsilon values (proposed `2/255`, `4/255`, `8/255`), 5–10 PGD
   steps as a pilot, and compare with sign-of-gradient FGSM and random noise.
5. Optimize using a fixed neutral or supportive prompt, then reuse the same
   image perturbation in all three evaluation contexts.
6. Check visual perceptibility and clipping of images, gradient magnitudes,
   attack effectiveness on the *source* objective, and peak MPS memory.
7. If gradients through SmolVLM2 are unsupported or too slow, use a small CLIP
   source model with PyTorch gradients and evaluate zero-gradient transfer to
   SmolVLM2. Report this as a **different transfer threat model**.

## Reporting requirements

- Model identifier + resolved revision, processor and package versions, dtype.
- macOS/PyTorch versions, hardware, memory, inference/attack timing.
- Exact prompts, preprocessing/resizing, and candidate scoring method.
- Clean accuracy before discussing attacks. Treat attacks on initially wrong
  items separately from attack success on initially correct items.
- Per-task and per-context breakdowns; no averaging away catastrophic slices.
- Explicit limitations: synthetic domain, small sample, potential data overlap
  between tasks from shared images, one primary model, non-independent tasks.
- Related research and what is or is not original in the experiment.

