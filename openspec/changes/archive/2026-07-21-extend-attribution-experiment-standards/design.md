## Context

The existing global attribution standard defines the core player set, AUTH-focus Attention shift, AUTH boundaries, and SPECIAL visualization treatment. This extension adds invariants that prevent future experiments from drifting in value functions, coalition construction, span handling, success-label semantics, and summary aggregation.

## Goals / Non-Goals

**Goals:**
- Keep `attention_shift` semantically stable and separate from attack-region dominance.
- Make Shapley values comparable by fixing the value function and masking protocol.
- Make region assignment auditable through non-overlapping multi-spans and explicit fixed-context treatment.
- Make outputs and summaries self-describing through denominator-aware Attention fields and target-scope labels.

**Non-Goals:**
- Introducing new attribution algorithms.
- Requiring regeneration-based Shapley or generation-time Attention analysis.
- Changing official dataset success labels.

## Decisions

- Use `attention_shift` only for the AUTH-focus threshold test; use a separate diagnostic such as `attention_attack_dominant` for `ATTACK > AUTH + FACT`.
- Use teacher-forced mean log probability as the Shapley value function because it explains the original behavior without generation randomness and avoids length bias from summed log probabilities.
- Preserve prompt structure across coalitions using embedding-level masking or a behaviorally equivalent mechanism; fixed context stays visible and non-player.
- Represent player regions as lists of token spans and validate that spans do not overlap.
- Emit denominator-specific Attention fields so raw, prompt-normalized, and player-normalized scores cannot be confused.
- Require summaries to group by `target_scope`, keeping action-wide attribution separate from tool-name or argument-only diagnostics.

## Risks / Trade-offs

- Older result files may not contain all recommended fields. Mitigation: visualization and analysis code can preserve backward-compatible fallbacks, but new outputs must follow the standard.
- Tokenization can create boundary-spanning tokens. Mitigation: span validation should prefer non-overlap, even if that means dropping ambiguous boundary tokens from a player region.
- Dataset labels may disagree with parsed actions. Mitigation: preserve official labels and record mismatch diagnostics instead of deriving labels from attribution.
