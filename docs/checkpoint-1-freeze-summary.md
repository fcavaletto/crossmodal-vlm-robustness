# Checkpoint 1 — freeze summary

**Date:** 2026-11-05  
**Branch:** `cursor/checkpoint-1-freeze-dabf`  
**Base tip:** `cursor/phase-d-dev-eval-dabf` @ `17d64b2`  
**Action:** Documentation freeze only — **no held-out access**, no Phase E runs, no merge to `main`.

## Federico decisions encoded

| ID | Decision | How encoded |
| --- | --- | --- |
| **D1(c)** | Report **both** DiDs with original scientific status | `PROTOCOL.md` §2: misleading−supportive = **preregistered primary** (negative color ≠ amplification; floor mandatory); neutral−supportive = **strongest secondary** (context changes vulnerability; answer-copying caveat; **not** a validated defense). No retrospective hypothesis rewrite. |
| **D2** | Freeze Phase D **as-run** | Full ε grid **{2,4,8}/255**, PGD 7, α=ε/4, opt@neutral, no-split, leading-space tokens. Clarified that “ε ∈ 8/255” alone would be a typo vs as-run set. Ceiling at ε=2/255 documented as limitation; **no retuning**. |

## Files touched

| File | Role |
| --- | --- |
| `docs/PROTOCOL.md` | Frozen human-readable protocol (replaces draft) |
| `docs/freeze-config.md` | Machine-readable pins |
| `docs/checkpoint-1-freeze-summary.md` | This summary |

## Explicit non-actions

- Held-out test **not** opened or evaluated  
- No new attacks / ε search / parser or prompt edits  
- No README portfolio rewrite (deferred)  
- No merge to `main`

## Next step (requires Federico go-ahead)

**Checkpoint 2:** one-shot held-out evaluation under this freeze (`PROTOCOL.md` §12), then technical report — still without post-hoc retuning.
