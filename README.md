# When text and pixels disagree: a small VLM stress test

Vision–language models read an image *and* a sentence. Suppose the sentence is wrong about what is in the picture. Does that make the model *easier* to fool with a carefully chosen pixel perturbation than when the sentence tells the truth?

That is the question this repository answers—on a deliberately tiny, fully labeled synthetic domain, with a frozen open-weight model, and with the answer allowed to be “no.”

We use [`SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) (revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`) on Apple MPS. Each image is a simple colored shape. The model sees one of three short contexts—**neutral**, **supportive** (true label), or **misleading** (false label)—then answers a one-word color or shape question. We craft a white-box L∞ image attack once under the neutral prompt, then score the *same* perturbed pixels under all three contexts. The primary contrast is a difference-in-differences (DiD) on *additional* error: misleading minus supportive. Full protocol: [`docs/PROTOCOL.md`](docs/PROTOCOL.md). Full write-up: [`reports/technical_report.md`](reports/technical_report.md).

## What happened on held-out data

On the sealed **32**-image test split, the primary DiD for PGD is **negative for color** at every frozen budget ε ∈ {2,4,8}/255. At ε=8/255 the estimate is **−0.500** (95% CI [−0.688, −0.344]). That does **not** support the amplification claim.

Two qualifications sit next to that number. First, clean accuracy under misleading color is already a **floor**—**4/32 = 12.5%** on test (development was **6/64 ≈ 9.4%**)—so there is little room for attacks to *add* misleading errors. Second, a **secondary** contrast (neutral − supportive) is positive for color: context still changes vulnerability under a fixed attack. That secondary signal is **not** a validated defense; supportive text restates the true label in a form the model can copy, and we have no mechanistic evidence beyond the paired error contrast.

![Held-out PGD DiD (bootstrap CIs)](reports/figures/phase-e/pgd_did_bootstrap.png)

For each task, the plot shows image-level DiD estimates versus ε. The mis−sup series below zero for color means *less* additional attack-caused error under misleading than under supportive—the opposite of amplification. The neu−sup series above zero means more additional error under neutral than supportive.

![Held-out robust accuracy vs ε (color)](reports/figures/phase-e/accuracy_vs_eps_color.png)

Curves are post-attack accuracy by context; grey lines mark clean accuracy. Supportive stays higher under PGD; misleading starts near the floor.

## What this shows, and what it does not

This repository shows a reproducible no-split MPS attack stack on SmolVLM2, a freeze-then-held-out workflow, and a negative primary result that survives the sealed test split—including floor effects, invalid-rate structure, and controls where noise is near-null and FGSM is much weaker than PGD.

It does **not** show that misleading text amplifies PGD relative to supportive text on this benchmark. It does not show that supportive captions are a general defense. It does not speak to real-world VLM safety rates, transfer attacks, or novelty relative to large multimodal benchmarks.

The analysis hierarchy (primary mis−sup; secondary neu−sup) was **frozen after Phase D development and before held-out**—not independently preregistered before exploration. We kept the original question and reported the negative primary finding rather than rewriting the story. Historical Phase A–E logs remain immutable.

Other limits worth naming up front: small \(n\) (32 test / 64 development), many reported cells without formal multiplicity correction, synthetic stimuli only, some non-monotonic attack-success rates versus ε, and no mechanistic evidence for why supportive text helps. Details live in the technical report §9.

## Run the checks locally (MacBook Air M4, 24 GB; PyTorch MPS)

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,plot]'
python -m pytest -q
```

On a fresh clone you can re-check the published statistics **without a GPU** and **without** gitignored `results/*.jsonl` files:

```bash
python -m crossmodal_robustness.verify_published
```

That command rebuilds accuracy, ASR, additional error, and DiD (with bootstrap CIs) from
`reports/artifacts/*_predictions_compact.jsonl` and compares them to
`reports/phase-e-heldout-metrics.json` and `reports/phase-d-full64-metrics.json`.
Exit code 0 means the published numbers match.

Prompt dry-run (no weights):

```bash
python -m crossmodal_robustness.evaluate --dry-run --task both --max-examples 2
```

Re-running held-out **inference** is a multi-hour one-shot under the freeze; see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Do not retune after seeing test numbers.

## Where to read next

Start with [`docs/BEGINNER_WALKTHROUGH.md`](docs/BEGINNER_WALKTHROUGH.md) if DiD or “floor” is new.
Then [`reports/technical_report.md`](reports/technical_report.md) for math and full tables,
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for exact commands and checksums, and
`reports/phase-*-verification.md` for chronological lab logs.

## Dataset

![Synthetic contact sheet](data/synthetic_v1/preview.png)

96 original 256×256 PNGs (4 colors × 4 shapes × 6 variants); 64 development / 32 held-out;
MIT-licensed procedural drawings; manifest with SHA-256 hashes. This is a controlled probe,
not a claim about natural imagery.

## Document map

| Doc | Path |
| --- | --- |
| Technical report | [`reports/technical_report.md`](reports/technical_report.md) |
| Frozen protocol | [`docs/PROTOCOL.md`](docs/PROTOCOL.md) |
| Freeze pins | [`docs/freeze-config.md`](docs/freeze-config.md) |
| Reproducibility | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| Beginner walkthrough | [`docs/BEGINNER_WALKTHROUGH.md`](docs/BEGINNER_WALKTHROUGH.md) |
| Consolidation / tip PR | [`docs/CONSOLIDATION.md`](docs/CONSOLIDATION.md) |
| Phase E verification | [`reports/phase-e-verification.md`](reports/phase-e-verification.md) |
| Artifact checksums + compact predictions | [`reports/artifacts/`](reports/artifacts/) |

## License

Source and synthetic images: [MIT](LICENSE). Third-party model weights follow the model owner’s terms.
