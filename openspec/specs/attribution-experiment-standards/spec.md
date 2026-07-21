# attribution-experiment-standards Specification

## Purpose
Define shared Attention and Shapley attribution semantics for prompt-injection experiments across QA and Agent-tool datasets, so future experiments use consistent player definitions, Attention-shift criteria, AUTH boundaries, SPECIAL diagnostics, and visualization normalization.
## Requirements
### Requirement: Attention Shift Uses AUTH Focus
Attention-shift detection SHALL be defined as the AUTH attention focus score being less than or equal to a threshold. The default threshold MUST be `0.5`. The AUTH focus score SHALL be computed over the player regions only as `AUTH / (AUTH + FACT + ATTACK)`.

#### Scenario: Detect shift when AUTH focus is below threshold
- **WHEN** an experiment computes player attention scores with `AUTH=0.3`, `FACT=0.5`, and `ATTACK=0.2`
- **THEN** it SHALL compute `auth_focus_score=0.3`
- **THEN** it SHALL set `attention_shift=true` when the threshold is `0.5`

#### Scenario: Do not detect shift when AUTH focus is above threshold
- **WHEN** an experiment computes player attention scores with `AUTH=0.6`, `FACT=0.2`, and `ATTACK=0.2`
- **THEN** it SHALL compute `auth_focus_score=0.6`
- **THEN** it SHALL set `attention_shift=false` when the threshold is `0.5`

### Requirement: Shapley Player Set Is Fixed
Shapley attribution experiments SHALL use exactly three players named `AUTH`, `FACT`, and `ATTACK`. Fixed prompt infrastructure and diagnostic regions MUST NOT be introduced as Shapley players.

#### Scenario: Construct Shapley coalitions
- **WHEN** an experiment computes Shapley values for prompt-injection attribution
- **THEN** it SHALL enumerate coalitions over `AUTH`, `FACT`, and `ATTACK`
- **THEN** it SHALL NOT include `SPECIAL`, tool schemas, fixed templates, or formatting scaffolds as Shapley players

### Requirement: AUTH Excludes Fixed And Nonsemantic Context
The AUTH region SHALL represent authorized task content only. It MUST exclude fixed templates, system or developer scaffolding, tool schemas, ReAct/output-format instructions, static prompt labels, and meaningless or control-only characters.

#### Scenario: Build AUTH for agent-tool prompts
- **WHEN** an agent-tool experiment builds regions for a prompt containing system text, tool schemas, ReAct format instructions, user instruction, and tool observations
- **THEN** the AUTH region SHALL include the authorized user instruction
- **THEN** the AUTH region SHALL exclude system text, tool schemas, ReAct format instructions, tool observations, attacker instruction text, and control-only tokens

### Requirement: SPECIAL Is Retained But Not Used For Player Normalization
Experiment results SHALL retain `SPECIAL` as a diagnostic fixed-context region. Visualizations comparing player attention SHALL internally normalize only `AUTH`, `FACT`, and `ATTACK`; `SPECIAL` MAY be shown separately as prompt-normalized diagnostic context.

#### Scenario: Visualize player attention with SPECIAL retained
- **WHEN** results contain attention scores for `AUTH`, `FACT`, `ATTACK`, and `SPECIAL`
- **THEN** the primary visualization SHALL compare `AUTH`, `FACT`, and `ATTACK` using `AUTH + FACT + ATTACK` as the denominator
- **THEN** the visualization SHALL NOT include `SPECIAL` in the player-normalized denominator
- **THEN** the visualization SHALL preserve `SPECIAL` in a separate diagnostic display or details panel

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

