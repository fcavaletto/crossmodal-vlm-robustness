# PR consolidation proposal (editorial; no merges performed)

**Date:** 2026-11-06  
**Phase F only — recommendations for Federico; agents must not merge or close PRs.**

## Recommended final reviewable tip

| Item | Recommendation |
| --- | --- |
| Branch | **`cursor/phase-f-docs-dabf`** (branched from Phase E tip `f3dd0bb`) |
| Why | Contains frozen protocol lineage + Phase E held-out verification + Phase F report/README/repro artifacts |
| Merge target (when approved) | `main`, preferably as **one** tip PR |

Open historical PRs (#1 Phase A … #6 Checkpoint 1, #7 Phase E) are intermediate lineage. Once the tip is accepted, prefer **not** merging them separately; leave them open or supersede with comments pointing at the tip (Federico’s call).

## Suggested PR hygiene (optional)

- Update PR #7 body to note Phase E **completed** and that Phase F docs live on `cursor/phase-f-docs-dabf`.  
- Add a comment on #1–#6: “superseded by tip; do not merge alone.”  
- Do **not** force-push, squash across unrelated history, or close without Federico.

## Clean structure (recommended tree for readers)

```text
README.md                          # public entry
docs/PROTOCOL.md                   # frozen protocol (immutable intent)
docs/freeze-config.md
docs/REPRODUCIBILITY.md
docs/BEGINNER_WALKTHROUGH.md
docs/pre-test-addendum-2026-11-05.md
reports/technical_report.md        # PDF-ready narrative
reports/phase-*-verification.md    # chronological lab logs
reports/phase-*-metrics.json
reports/artifacts/                 # checksums + compact predictions
reports/figures/phase-{d,e}/
```

Historical planning docs (`docs/phase-a-plan.md`, Checkpoint 0 audit copies) may remain for chronology; they are not the reader’s primary path.
