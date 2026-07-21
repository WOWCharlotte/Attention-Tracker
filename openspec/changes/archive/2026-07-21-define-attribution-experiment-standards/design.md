## Context

The project runs attribution experiments across QA datasets and Agent-tool datasets. The first-stage QA workflow defines Attention shift through an AUTH-focus detector, while later Agent-tool work introduced finer attack-region diagnostics. The global standard must keep the primary Attention-shift semantics consistent across stages while still retaining Agent-specific diagnostics such as `SPECIAL` and attack-region dominance.

## Goals / Non-Goals

**Goals:**
- Make `attention_shift` mean the same thing in every experiment: AUTH focus is at or below a threshold, default `0.5`.
- Keep Shapley attribution limited to `AUTH`, `FACT`, and `ATTACK`.
- Keep fixed prompt infrastructure out of `AUTH` and Shapley players.
- Preserve `SPECIAL` in outputs while excluding it from player-normalized visual comparisons.

**Non-Goals:**
- Redefining attack success labels.
- Removing `SPECIAL` from raw experiment outputs.
- Replacing existing token-level visual diagnostics.

## Decisions

- Use `attention_shift` for the stage-aligned AUTH-focus criterion. Any metric that checks whether `ATTACK > AUTH + FACT` must use a separate name such as `attention_attack_dominant`.
- Compute AUTH focus from player regions only: `AUTH / (AUTH + FACT + ATTACK)`. This mirrors the first-stage instruction-vs-data detector while excluding Agent fixed context from AUTH and the denominator.
- Treat fixed templates, tool schemas, ReAct formatting, static labels, and meaningless/control-only tokens as non-player context. These can be retained as `SPECIAL` diagnostics but cannot contribute to `AUTH`, `FACT`, `ATTACK`, or Shapley coalitions.
- Visualizations should make player-normalized `AUTH/FACT/ATTACK` the primary comparison. Prompt-normalized views including `SPECIAL` should be secondary diagnostics, preferably collapsed by default when screen space is limited.

## Risks / Trade-offs

- Existing result files may contain older `attention_shift_attack` semantics. Mitigation: new scripts should emit `attention_shift_basis`, `auth_focus_score`, `threshold`, and separate `attention_attack_dominant` fields.
- `SPECIAL` can dominate raw attention in Agent prompts. Mitigation: keep raw and prompt-normalized `SPECIAL` diagnostics, but use player-normalized scores for comparisons among Shapley players.
- AUTH boundaries differ by dataset format. Mitigation: each experiment must explicitly validate that AUTH excludes fixed infrastructure and attacker/data spans.
