# Model Designer design

## Goal

Add an **interactive model designer** to the TUI so an operator can configure a model family and its hyperparameters visually, validate the resulting model configuration, preview the normalized payload, and hand the result back to the existing **run designer** without manually editing JSON or YAML.

This is the sibling to the run designer. The run designer owns the full training config; the model designer specializes in the `model` section and model-specific defaults.

## Why this exists

The run designer already lets the user edit `model.kind`, `device`, and raw `model_params_json`. That works, but it is still annoyingly wizard-adjacent in the wrong way:
- users must know parameter names up front,
- they can create invalid parameter shapes by typing raw JSON,
- switching model families does not guide them toward relevant knobs,
- parameter discoverability is weak.

A model designer should make model configuration explicit and interactive.

## Scope

### In scope for v1
- dedicated **model designer lane** in the TUI
- consistent naming of the existing **run designer** lane in code/docs/help text
- load defaults for each supported model family
- clone model settings from the currently selected config or current run-designer draft
- edit a curated set of model-specific fields through form widgets instead of raw JSON only
- preview normalized `model:` YAML
- validate model kind, device, and parameter values before applying
- apply the designed model back into the run designer draft
- show clear status feedback after preview, validation, and apply actions
- testable pure helper layer for draft conversion and validation
- docs and help text updates

### Explicitly out of scope for v1
- launching a training run directly from the model designer
- async background tuning or sweeps
- arbitrary plugin-defined models
- full hyperparameter search UI
- nested modal dialogs or popup windows
- schema generation from type metadata

## Naming and lane ownership

This design must avoid the word "designer" meaning two different things.

### Canonical vocabulary
- **run designer** = full `TrainingConfig` editor and launcher
- **model designer** = specialized `model` workbench that feeds the run designer

### Mode selector
The top-level mode selector becomes:
- `runs`
- `configs`
- `run-designer`
- `model-designer`

Internally, code should stop using a generic ambiguous `designer` label when it really means the run designer.

### Buttons and help text
Toolbar/button language should explicitly say:
- `Design Run`
- `Design Model`

Help and action catalog text should use the same phrases.

## UX overview

### Navigation model
Keyboard:
- `tab` cycles across all four lanes
- `n` opens the run designer directly
- `b` opens the model designer directly

Toolbar:
- keep **Design Run**
- add **Design Model** next to it

### Relationship with the run designer
The model designer is not a separate artifact authoring system. It is a focused workbench for the `model` slice.

The intended flow is:
1. open run designer
2. switch to model designer when model work is needed
3. adjust model family and parameters with guided controls
4. preview/validate
5. **Apply To Run Draft**
6. return to run designer for dataset/evaluation/tracking choices
7. save/launch from the run designer

That keeps ownership clean:
- run designer owns full `TrainingConfig`
- model designer owns `ModelConfig` editing ergonomics

## Supported model families

Use exactly the model kinds already supported by config/modeling code:
- `sklearn_logreg`
- `sklearn_random_forest`
- `sklearn_hist_gradient_boosting`
- `pytorch_mlp`

## Existing mismatch to fix before implementation

The current TUI exposes an unsupported XGBoost-ish option in one path even though the config/modeling layers only support the four families above.

Before implementation, centralize supported model-family options from the config layer so the run designer and model designer cannot drift.

As part of that change:
- remove the unsupported option from the current TUI surface,
- make the centralized list the required source for both run-designer and model-designer selectors.

This is a real blocker, not a cosmetic note.

## v1 form design

### Common controls
Every model draft includes:
- `model kind` select
- `device` select
- `parameter preset source` status text: `defaults`, `cloned-from-config`, or `cloned-from-run-draft`

Device options:
- `auto`
- `cpu`
- `cuda`

### Active-family-only fields
To avoid turning the details pane into a spaghetti garden, only the active family’s fields should be visible.

#### sklearn_logreg
- `c` input
- `max_iter` input

#### sklearn_random_forest
- `n_estimators` input
- `max_depth` input, with blank meaning `None`
- `min_samples_leaf` input

#### sklearn_hist_gradient_boosting
- `learning_rate` input
- `max_iter` input
- `max_depth` input, blank meaning `None`
- `min_samples_leaf` input

#### pytorch_mlp
- `hidden_dims` input as comma-separated integers
- `epochs` input
- `batch_size` input
- `learning_rate` input
- `dropout` input

### Secondary panes

#### Left list pane
When in `model-designer` mode, the left list shows **model family templates** only:
- Logistic Regression
- Random Forest
- HistGradientBoosting
- PyTorch MLP

Selecting a family in the list changes the highlighted template target only. It does **not** overwrite the current draft.

#### Overview pane
Show:
- selected model family template
- current draft family
- whether the draft is dirty
- source of the current draft
- quick reminder of available actions

#### Details pane
Show one of:
- normalized model YAML preview, or
- validation errors when preview/validation fails

#### Task/status pane
Show success/failure messages for preview, validate, clone, and apply actions.

## Buttons and actions

### Buttons
- `Load Model Defaults`
- `Clone From Config`
- `Clone From Run Draft`
- `Preview Model YAML`
- `Validate Model Draft`
- `Apply To Run Draft`

### Hotkeys
- `b` open model designer lane
- `y` preview model YAML
- `u` validate model draft
- `o` apply model draft to the run designer

Hotkeys should mirror the existing TUI style: helpful, but not required for basic use.

## Synchronization contract with the run designer

This section is mandatory because state drift is otherwise guaranteed.

### On opening the model designer
Initialization order:
1. if the run designer has an active draft, hydrate the model designer from that draft’s model slice
2. else if a config is selected in the config browser, hydrate from that config
3. else hydrate from project defaults for `DEFAULT_MODEL_KIND`

### On editing inside the model designer
- edits affect only the model-designer draft
- they do **not** mutate the run-designer draft immediately
- the model-designer draft gets its own dirty flag

### On `Apply To Run Draft`
If validation succeeds:
- update only `model_kind`, `device`, and `model_params_json` in the run-designer draft
- refresh the run-designer preview text
- keep dataset/evaluation/tracking fields unchanged
- mark the run-designer draft dirty
- mark the model-designer draft clean
- show success feedback naming the applied model family

If there is no run-designer draft for some reason:
- create one from run-designer defaults
- then apply the model slice into it

### After apply
The model designer does not live-sync with subsequent run-designer edits. Sync is explicit, not magical.

Rules:
- re-opening model designer rehydrates from the latest run-draft model slice
- if the user remains in model designer after apply and then the run draft changes elsewhere, the model designer keeps its local draft until explicitly reopened or re-cloned

That keeps behavior understandable and testable.

## Family-switch behavior

This is another anti-footgun rule.

- moving selection in the left template list never mutates the current form
- clicking `Load Model Defaults` loads defaults for the currently highlighted family
- clicking `Clone From Config` or `Clone From Run Draft` replaces the draft with the chosen source

Per-family edits are not silently preserved in background hidden caches for v1. The behavior stays explicit:
- current draft is one concrete model draft
- explicit clone/default actions replace it
- selection alone does not replace it

This is simpler than maintaining four subdraft caches and reduces hidden state.

## Module split

Create a new module dedicated to model-designer helpers instead of cramming more cross-cutting logic into `tui.py`.

Proposed module:
- `src/bc_mlops_showcase/model_designer.py`

Core dataclasses:
- `ModelDesignerDraft`
  - `model_kind: str`
  - `device: str`
  - `logreg_c: str`
  - `logreg_max_iter: str`
  - `rf_n_estimators: str`
  - `rf_max_depth: str`
  - `rf_min_samples_leaf: str`
  - `hgb_learning_rate: str`
  - `hgb_max_iter: str`
  - `hgb_max_depth: str`
  - `hgb_min_samples_leaf: str`
  - `mlp_hidden_dims: str`
  - `mlp_epochs: str`
  - `mlp_batch_size: str`
  - `mlp_learning_rate: str`
  - `mlp_dropout: str`
  - `source_name: str | None = None`

- `ModelDesignerValidationResult`
  - `ok: bool`
  - `errors: list[str]`
  - `resolved_model: ModelConfig | None`

Functions:
- `build_default_model_designer_draft(model_kind: str = DEFAULT_MODEL_KIND) -> ModelDesignerDraft`
- `build_model_designer_draft_from_model_config(model: ModelConfig, source_name: str | None = None) -> ModelDesignerDraft`
- `build_model_designer_draft_from_training_config(config: TrainingConfig, source_name: str | None = None) -> ModelDesignerDraft`
- `model_designer_draft_to_model_config(draft: ModelDesignerDraft) -> ModelConfig`
- `validate_model_designer_draft(draft: ModelDesignerDraft) -> ModelDesignerValidationResult`
- `render_model_designer_preview_text(draft: ModelDesignerDraft) -> str`
- `apply_model_designer_draft_to_run_draft(model_draft: ModelDesignerDraft, run_draft: DesignerDraft) -> DesignerDraft`

## Validation rules

Validation should be explicit and deterministic.

### Common rules
- `model_kind` must be supported
- `device` must be one of `auto`, `cpu`, `cuda`

### sklearn_logreg
- `c` must be `> 0`
- `max_iter` must be integer `>= 1`

### sklearn_random_forest
- `n_estimators` integer `>= 1`
- `max_depth` blank => `None`, else integer `>= 1`
- `min_samples_leaf` integer `>= 1`

### sklearn_hist_gradient_boosting
- `learning_rate` float `> 0`
- `max_iter` integer `>= 1`
- `max_depth` blank => `None`, else integer `>= 1`
- `min_samples_leaf` integer `>= 1`

### pytorch_mlp
- `hidden_dims` must parse into a non-empty list of positive ints
- `epochs` integer `>= 1`
- `batch_size` integer `>= 1`
- `learning_rate` float `> 0`
- `dropout` float in `[0, 1)`

Validation should report user-facing messages, not raw Python tracebacks.

## Validation boundary, not just TUI sugar

User-friendly validation belongs in `model_designer.py`, but canonical enforcement should not exist only in the TUI.

Implementation should also add or reuse a lower-level validation path so that unsupported devices/invalid model settings cannot sneak in through other code paths.

At minimum, the implementation should ensure:
- supported model kind validation remains canonical in the config layer
- explicit device validation exists below the TUI layer
- model-parameter normalization used by the model designer can be trusted by non-TUI callers too

### CUDA policy
For v1:
- `device=cuda` is syntactically valid even if CUDA is unavailable on the current machine
- the UI may warn in status/help text, but should not reject the config purely because this terminal session lacks CUDA

That keeps the designer focused on configuration correctness rather than current-host scheduling constraints.

## Interaction with config defaults

The model designer must stay aligned with `DEFAULT_MODEL_PARAMS` in `config.py`.

Rule:
- default values come from the config layer
- the model designer may present them in specialized fields
- conversion back to `ModelConfig.params` must preserve the existing canonical parameter names

Do not introduce a second independent source of truth for defaults.

## TUI implementation notes

### Visibility
Like the run designer, the model designer should be mounted once and shown/hidden based on `self.mode`.

### Sync guards
Add a dedicated guard like:
- `self._is_syncing_model_designer`

This prevents widget-update loops while refreshing the form from draft state.

### Layout expectations
To keep the lane usable inside the existing TUI:
- always show common controls
- show only active-family parameter inputs
- use the existing details pane for preview/errors
- use the existing task/status pane for action feedback
- keep the left list dedicated to model families while in `model-designer` mode

The model designer should not attempt a denser multi-column parameter matrix in v1.

## Testing strategy

### Pure helper tests
Add a new test file:
- `tests/test_model_designer.py`

Cover:
- defaults per model family
- cloning from `ModelConfig`
- cloning from `TrainingConfig`
- converting draft -> `ModelConfig`
- invalid numeric validation cases
- invalid hidden_dims parsing
- blank max_depth -> `None`
- invalid device rejection
- apply-to-run-draft updates only the model slice
- preview rendering contains normalized YAML

### Interactive TUI tests
Extend:
- `tests/test_interactive_tui.py`

Cover at least:
- model designer lane opens from button/hotkey
- defaults load for selected family
- preview button updates details pane
- validate button surfaces success/failure
- apply button updates the run designer draft/model preview
- switching back to run designer preserves the applied model settings
- opening model designer from a dirty run draft hydrates from that draft
- list selection alone does not mutate drafts

### Regression tests
Re-run existing tests that protect:
- lazy import boundaries for dashboard command
- current run designer flow
- non-interactive dashboard rendering
- supported-model option consistency

## Documentation updates

Update:
- `README.md`
- `docs/source/usage.rst`
- generated API docs for the new module
- in-app help/catalog strings in `tui.py`

Docs must describe:
- how to open the model designer
- what it edits
- how it connects back to the run designer

## Suggested implementation order

1. design doc commit
2. fix/centralize supported-model option list
3. add pure `model_designer.py` helpers + tests
4. wire run-designer/model-designer synchronization rules
5. integrate the new lane into `tui.py`
6. update docs/generated docs

## Commit plan

1. design doc commit
2. model support/options cleanup + helper module + tests
3. TUI integration commit
4. docs/generated-docs commit

## Risks and tradeoffs

### Tradeoff: explicit fields vs generic schema
A generic schema-driven form would be more extensible, but it is heavier than needed for four known model families. v1 should stay explicit and readable.

### Risk: too much logic in `tui.py`
The current TUI file is already large. Mitigate by keeping parsing/validation/preview logic in `model_designer.py` and limiting `tui.py` to orchestration.

### Risk: run designer / model designer state drift
Mitigate by making `Apply To Run Draft` the explicit synchronization point and documenting rehydration rules clearly.

## Ready-for-implementation criteria

Implementation can begin once the reviewer agrees that:
- the run designer and model designer are clearly named and separated,
- silent overwrite behavior is prevented,
- validation rules are concrete,
- supported model-family drift is addressed,
- pure helpers own model parsing/normalization,
- the handoff back into the run designer is explicit and testable.