## ADDED Requirements

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
