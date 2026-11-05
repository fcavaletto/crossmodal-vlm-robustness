"""Process-stable seed derivation (no reliance on PYTHONHASHSEED / ``hash()``)."""

from __future__ import annotations

import hashlib


def stable_seed_int(*parts: object, modulus: int = 2_147_483_647) -> int:
    """Derive a non-negative int seed from stringifiable parts via MD5.

    Unlike Python's built-in ``hash()``, this is stable across processes and
    interpreter invocations without setting ``PYTHONHASHSEED``.
    """
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return int(digest, 16) % modulus


def attack_noise_seed(
    *,
    base_seed: int,
    image_index: int,
    task: str,
    attack_name: str,
    epsilon: float,
) -> int:
    """Seed for equal-budget random L∞ noise for one (image, task, ε) row."""
    return stable_seed_int(
        base_seed,
        image_index,
        task,
        attack_name,
        f"{epsilon:.8f}",
        "linf_noise_v1",
        modulus=10_007,
    ) + base_seed
