"""Deterministic baseline inference on a small open-weight vision-language model.

This is baseline evaluation only. Adversarial VLM input-gradient integration is
deliberately NOT included until a smoke test on the target Mac validates it.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .dataset import build_prompt, load_manifest, parse_answer

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def select_device(preferred: str):
    import torch

    if preferred == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if preferred == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available; try --device cpu")
    return preferred


def load_model(model_id: str, device: str, *, revision: str | None = None):
    """Load processor + model; optionally pin and verify a HF revision.

    When ``revision`` is set, both the processor and the model are loaded from
    that commit. Verification prefers object ``_commit_hash`` when present;
    otherwise confirms via ``huggingface_hub.model_info(...).sha`` that the
    requested revision resolves to the frozen commit (Transformers 5.x may omit
    ``_commit_hash`` on this checkpoint).
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    pretrained_kwargs = {}
    if revision is not None:
        pretrained_kwargs["revision"] = revision
        hub_sha = _hub_revision_sha(model_id, revision)
        _assert_revision_match("hub", hub_sha, revision)

    processor = AutoProcessor.from_pretrained(model_id, **pretrained_kwargs)
    # Prefer `dtype=` (Transformers >=5); fall back for older releases.
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float32,
            attn_implementation="eager",
            **pretrained_kwargs,
        ).to(device)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            **pretrained_kwargs,
        ).to(device)
    model.eval()
    if revision is not None:
        model_commit = _resolved_commit_hash(model)
        processor_commit = _resolved_commit_hash(processor)
        if model_commit is not None:
            _assert_revision_match("model", model_commit, revision)
        if processor_commit is not None:
            _assert_revision_match("processor", processor_commit, revision)
        processor._frozen_revision = revision  # type: ignore[attr-defined]
        model.config._frozen_revision_verified = revision
        model.config._frozen_revision_hub_sha = hub_sha
    return processor, model


def _hub_revision_sha(model_id: str, revision: str) -> str:
    from huggingface_hub import model_info

    info = model_info(model_id, revision=revision)
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"huggingface_hub returned no sha for {model_id}@{revision}")
    return str(sha)


def _resolved_commit_hash(obj) -> str | None:
    """Best-effort HF commit hash from a loaded model or processor."""
    for candidate in (
        getattr(obj, "config", None),
        obj,
        getattr(obj, "tokenizer", None),
        getattr(obj, "image_processor", None),
    ):
        if candidate is None:
            continue
        commit = getattr(candidate, "_commit_hash", None)
        if commit:
            return str(commit)
    return None


def _assert_revision_match(label: str, got: str | None, wanted: str) -> None:
    if got is None:
        raise RuntimeError(
            f"Loaded {label} has no commit hash; cannot verify revision {wanted}"
        )
    if not str(got).startswith(wanted[:12]) and str(got) != wanted:
        raise RuntimeError(f"{label} revision mismatch: wanted {wanted}, got {got}")


def run_inference(
    processor,
    model,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    *,
    do_image_splitting: bool = True,
    image: Image.Image | None = None,
):
    """Perform generation through the official Transformers multimodal API.

    ``do_image_splitting`` must match adversarial preprocessing when evaluating
    attacked images. Pass a PIL ``image`` to skip reloading from disk (used for
    in-memory perturbed tensors converted once at the eval boundary).
    """
    import torch

    if image is None:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
    else:
        image = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_image_splitting": do_image_splitting},
    )
    inputs = {name: value.to(model.device) if hasattr(value, "to") else value
              for name, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
    # SmolVLM2 is a decoder-only model: generated IDs include the prompt prefix.
    prompt_size = inputs["input_ids"].shape[-1]
    return processor.batch_decode(generated[:, prompt_size:], skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic_v1/manifest.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--split", default="development", choices=["development", "test"])
    parser.add_argument("--task", default="color", choices=["color", "shape", "both"])
    parser.add_argument("--conditions", nargs="+", default=["neutral", "supportive", "misleading"],
                        choices=["neutral", "supportive", "misleading"])
    parser.add_argument("--max-examples", type=int, default=8,
                        help="Number of images, not image/task/condition evaluations")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/baseline.jsonl"))
    parser.add_argument(
        "--do-image-splitting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SmolVLM image splitting. Phase C uses --no-do-image-splitting consistently.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect prompts without downloading a model")
    args = parser.parse_args()

    records = load_manifest(args.manifest, split=args.split)
    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    if args.max_examples:
        records = records[: args.max_examples]
    tasks = ["color", "shape"] if args.task == "both" else [args.task]
    jobs = [(record, task, condition) for record in records for task in tasks
            for condition in args.conditions]

    if args.dry_run:
        print(f"Dry run: {len(records)} images, {len(jobs)} prompts; no model loaded.")
        for record, task, condition in jobs[:6]:
            print(f"\n[{record['id']} | {task} | {condition}]\n{build_prompt(record, task, condition)}")
        return

    import torch
    import transformers

    device = select_device(args.device)
    print(
        f"Loading {args.model} on {device}; {len(jobs)} evaluations; "
        f"do_image_splitting={args.do_image_splitting}",
        flush=True,
    )
    processor, model = load_model(args.model, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = defaultdict(lambda: [0, 0])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (record, task, condition) in enumerate(jobs, start=1):
            prompt = build_prompt(record, task, condition)
            start = time.perf_counter()
            answer = run_inference(
                processor,
                model,
                Path(record["absolute_image"]),
                prompt,
                args.max_new_tokens,
                do_image_splitting=args.do_image_splitting,
            )
            elapsed = time.perf_counter() - start
            parsed = parse_answer(answer, task)
            correct = parsed == record[task]
            summary[(task, condition)][0] += int(correct)
            summary[(task, condition)][1] += 1
            row = {
                "run_id": run_id,
                "image_id": record["id"],
                "image_sha256": record["sha256"],
                "split": record["split"],
                "task": task,
                "condition": condition,
                "expected": record[task],
                "claimed_context": record[f"wrong_{task}"] if condition == "misleading" else
                                   (record[task] if condition == "supportive" else None),
                "prompt": prompt,
                "raw_answer": answer,
                "parsed_answer": parsed,
                "correct": correct,
                "seconds": round(elapsed, 3),
                "attack": "none",
                "do_image_splitting": args.do_image_splitting,
                "model": args.model,
                "device": device,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "python_version": platform.python_version(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(f"{index}/{len(jobs)} {record['id']} {task} {condition}: "
                  f"{answer!r} ({'correct' if correct else 'incorrect/invalid'})", flush=True)

    print("\nSummary (strict exact-answer accuracy):")
    for (task, condition), (correct, total) in sorted(summary.items()):
        print(f"{task:5s} {condition:10s} {correct}/{total} = {correct / total:.1%}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

