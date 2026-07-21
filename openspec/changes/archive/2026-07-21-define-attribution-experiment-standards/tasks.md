## 1. Specification

- [x] 1.1 Define the global Attention-shift criterion as AUTH focus at or below threshold `0.5`.
- [x] 1.2 Define the global Shapley player set as `AUTH`, `FACT`, and `ATTACK`.
- [x] 1.3 Define AUTH boundaries to exclude fixed templates, tool schemas, formatting scaffolds, and meaningless/control-only characters.
- [x] 1.4 Define SPECIAL retention in experiment outputs and exclusion from player-normalized visualization denominators.

## 2. Implementation Alignment

- [x] 2.1 Align InjecAgent Attention output fields with AUTH-focus Attention shift semantics.
- [x] 2.2 Preserve attack-region dominance as a separate diagnostic metric instead of the primary Attention-shift criterion.
- [x] 2.3 Align InjecAgent visualization with player-normalized `AUTH/FACT/ATTACK` comparison and separate SPECIAL diagnostics.
- [x] 2.4 Add tests covering AUTH-focus shift semantics, SPECIAL exclusion, and visualization normalization.
