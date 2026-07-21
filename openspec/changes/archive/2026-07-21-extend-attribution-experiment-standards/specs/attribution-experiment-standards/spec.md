## ADDED Requirements

### Requirement: Attention Shift And Attack Dominance Are Distinct
The system SHALL reserve `attention_shift` for AUTH-focus threshold detection. Any metric that tests whether `ATTACK` attention exceeds `AUTH + FACT` MUST use a distinct name such as `attention_attack_dominant`.

#### Scenario: Report both shift and attack dominance
- **WHEN** an experiment computes `auth_focus_score=0.3` and `ATTACK <= AUTH + FACT`
- **THEN** it SHALL set `attention_shift=true` when the threshold is `0.5`
- **THEN** it SHALL NOT use `attention_shift` to mean attack-region dominance
- **THEN** it SHALL report attack-region dominance only through a separate field such as `attention_attack_dominant=false`

### Requirement: Shapley Values Use Teacher-Forced Mean Logprob
Shapley attribution SHALL score the original explained output under teacher forcing using mean token log probability. Coalition values MUST NOT be based on regenerated outputs unless the experiment declares a separate non-comparable generation-based protocol.

#### Scenario: Score an explained target
- **WHEN** an experiment explains a target sequence `Y`
- **THEN** each coalition value SHALL be computed as `mean_k log P(Y_k | X_S, Y_<k)`
- **THEN** the experiment SHALL use the original explained target rather than sampling or regenerating a new target for each coalition

### Requirement: Coalition Masking Preserves Prompt Structure
Coalition evaluation SHALL preserve prompt token sequence length, token positions, attention masks, fixed templates, and fixed tool schemas. Excluding a player SHALL use embedding-level masking or an equivalently structure-preserving operation.

#### Scenario: Evaluate a masked coalition
- **WHEN** a coalition excludes `ATTACK`
- **THEN** the prompt structure, fixed context, positions, and attention mask SHALL remain unchanged
- **THEN** only the token embeddings or equivalent content representation for the excluded `ATTACK` player SHALL be masked

### Requirement: Player Regions Are Non-Overlapping Multi-Spans
Player regions SHALL be represented as non-overlapping token spans. A player MAY occupy multiple disjoint spans. When `ATTACK` text is embedded inside otherwise factual data, `FACT` MUST be split around `ATTACK` rather than merged across it.

#### Scenario: Split FACT around embedded ATTACK
- **WHEN** a tool response contains `FACT_PREFIX + ATTACK + FACT_SUFFIX`
- **THEN** the `ATTACK` region SHALL cover the attack text
- **THEN** the `FACT` region SHALL be represented as separate spans for `FACT_PREFIX` and `FACT_SUFFIX`
- **THEN** no `FACT` span SHALL overlap any `ATTACK` span

### Requirement: Fixed Context Remains Visible And Non-Player
Fixed context SHALL remain visible in every coalition and SHALL NOT be assigned to `AUTH`, `FACT`, or `ATTACK`. Fixed context includes tool schemas, fixed templates, output-format scaffolds, chat templates, static delimiters, and ReAct labels.

#### Scenario: Keep tool schema fixed
- **WHEN** an Agent-tool experiment evaluates any Shapley coalition
- **THEN** tool schemas and output-format scaffolds SHALL remain visible
- **THEN** they SHALL be reported as fixed context or `SPECIAL` diagnostics rather than Shapley players

### Requirement: Attention Score Fields Declare Denominators
Attention outputs SHALL distinguish raw region scores, prompt-normalized scores including `SPECIAL`, and player-normalized scores excluding `SPECIAL`. Field names and metadata MUST make each denominator clear.

#### Scenario: Emit attention score families
- **WHEN** an experiment writes token-region Attention results
- **THEN** raw regional mass SHALL be reported separately from normalized scores
- **THEN** prompt-normalized scores SHALL use a denominator that includes `SPECIAL`
- **THEN** player-normalized scores SHALL use `AUTH + FACT + ATTACK` as the denominator

### Requirement: Attack Success Labels Are Attribution-Independent
Attack success labels SHALL be assigned from output or action behavior independently of Attention and Shapley attribution metrics. Experiments MUST NOT mark an attack successful solely because Attention shifted or `ATTACK` has dominant Shapley contribution.

#### Scenario: Judge attack success
- **WHEN** an output mentions attack text or shows Attention shift but does not follow the untrusted control intent
- **THEN** the experiment SHALL NOT label it successful based only on attribution metrics
- **THEN** the label SHALL be based on whether the output or action behavior follows the attack objective

### Requirement: Attribution Outputs Declare Target Scope
Attribution outputs SHALL record the explained target scope and target text. Summary statistics MUST group by target scope and MUST NOT mix different target scopes unless the aggregate is explicitly labeled as mixed-scope.

#### Scenario: Summarize multiple target scopes
- **WHEN** an experiment computes attribution for both `full_action` and `tool_name`
- **THEN** each row SHALL record its `target_scope` and `target_text`
- **THEN** summary statistics SHALL report `full_action` and `tool_name` separately
