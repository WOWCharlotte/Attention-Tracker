## Why

Attention and Shapley analyses are reused across QA and Agent-tool experiments, but their shared semantics have drifted during implementation. This change establishes a global experiment contract so future work keeps Attention-shift detection, Shapley players, AUTH boundaries, SPECIAL handling, and visualization normalization aligned across phases.

## What Changes

- Define the canonical Attention-shift criterion as AUTH attention focus falling below a threshold, defaulting to `0.5`.
- Define the canonical Shapley player set as `AUTH`, `FACT`, and `ATTACK`.
- Require `AUTH` regions to exclude fixed templates, tool schemas, formatting scaffolds, and meaningless/control characters.
- Require experiment outputs to preserve `SPECIAL` as a diagnostic fixed-context region.
- Require visualizations to compare `AUTH`, `FACT`, and `ATTACK` using internal/player normalization, while showing `SPECIAL` only as retained diagnostic context.

## Capabilities

### New Capabilities
- `attribution-experiment-standards`: Shared semantics for Attention and Shapley attribution experiments across QA and Agent-tool datasets.

### Modified Capabilities

## Impact

- Affects experiment scripts that compute Attention shift, Shapley attribution, token regions, and summaries.
- Affects visualization scripts that display Attention region scores and dominance labels.
- No new runtime dependencies or external APIs are required.
