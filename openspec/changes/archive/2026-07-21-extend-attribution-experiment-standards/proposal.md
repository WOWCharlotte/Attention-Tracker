## Why

The existing attribution standards capture the core Attention-shift, player, AUTH, and SPECIAL rules. Additional experiment invariants should be globalized so future QA, InjecAgent, and AgentDojo extensions do not silently change target semantics, masking behavior, value functions, or aggregation rules.

## What Changes

- Separate AUTH-focus Attention shift from attack-region dominance by naming and output contract.
- Require Shapley values to use teacher-forced mean log probability over the original explained target.
- Require coalition masking to preserve prompt structure and fixed context.
- Require non-overlapping multi-span player regions.
- Require fixed context to remain visible in every coalition.
- Require clear raw, prompt-normalized, and player-normalized Attention field semantics.
- Require attack success labels to be assigned independently from attribution metrics.
- Require attribution outputs and summaries to declare and respect target scope.

## Capabilities

### New Capabilities

### Modified Capabilities
- `attribution-experiment-standards`: Add cross-phase requirements for naming, scoring, masking, spans, labels, normalization fields, target scope, and fixed-context handling.

## Impact

- Affects future experiment and visualization scripts that compute or report Attention/Shapley metrics.
- Affects output schema expectations for attribution JSONL and summary files.
- No new runtime dependencies or external APIs are required.
