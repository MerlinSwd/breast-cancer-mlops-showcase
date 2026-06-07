# Run Designer design

## Goal

Add a **run designer** to the interactive TUI so an operator can create, edit, validate, preview, save, and launch a training configuration without hand-editing YAML first.

The run designer should make it easy to:
- start from project defaults,
- clone an existing config,
- tweak common fields from a form-like UI,
- preview the normalized YAML before launch,
- save the config as a reusable file,
- launch training directly from the TUI,
- get status feedback about what was saved and what was launched.

## Non-goals

This first iteration will **not**:
- support arbitrary YAML editing inside the TUI,
- expose every possible future backend-specific parameter in bespoke widgets,
- stream live subprocess logs,
- allow editing completed run artifacts in place,
- introduce background job scheduling from the designer,
- attempt modal-heavy confirmation UX unless a simple status warning proves insufficient.

Those would be neat. So are dragons.

## Current context

The project already has:
- `TrainingConfig` dataclasses and YAML loading in `src/bc_mlops_showcase/config.py`
- config browsing in the TUI via `ConfigView` / `load_config_views(...)`
- in-TUI run actions (`validate`, `report`, `predict`, `retrain`) in `src/bc_mlops_showcase/tui.py`
- a single `train` CLI command in `src/bc_mlops_showcase/cli.py`
- several canonical config files under `configs/`

The current TUI can inspect configs, but cannot **author** one. The run designer fills that gap.

## UX overview

### Entry points

The operator can open the run designer in three ways:
1. press a new top-bar button: **Design Run**
2. press a new hotkey: `n`
3. while in config-browser mode, press a new button/hotkey to **Clone into Designer**

### Workspace model

The designer will live inside the existing command deck instead of launching a separate app.

It adds a third workspace lane:
- `runs`
- `configs`
- `designer`

### Designer layout

The first iteration should stay deliberately narrow and use a **two-pane layout**:

1. **Editor pane**
   - source/template buttons at the top
   - editable fields below

2. **Preview / feedback pane**
   - normalized YAML preview
   - validation errors
   - latest save/launch status

The source/template controls should be buttons above the form, not a whole extra panel. Trying to cram four independent panels into the current deck would be terminal maximalism at its least charming.

### Editor fields for v1

The first version should edit the common, high-value fields directly:
- experiment name
- config file name / slug
- random seed
- threshold
- dataset kind
- dataset path
- target column
- model kind
- device
- evaluation mode
- folds
- test size
- stratify
- tracking experiment name
- model params as editable JSON text

Fields like `drop_columns`, `positive_label`, and `tracking.uri` may still exist in the underlying config, but they do not need bespoke first-class widgets in v1. They can be added later once the flow is proven.

## Interaction model

### Buttons / menus first

Because the user explicitly asked for more buttons and menus, the designer should lean into explicit controls instead of relying on secret wizard keys.

Add:
- a toolbar button: `Design Run`
- designer-specific buttons:
  - `Load Defaults`
  - `Clone Config`
  - `Preview YAML`
  - `Validate Draft`
  - `Save Config`
  - `Launch Run`

Use Textual widgets such as:
- `Select` for finite-choice fields
- `Input` for scalar text / numbers
- `Checkbox` or `Switch` for booleans
- `TextArea` for model params JSON and YAML preview if available in the installed Textual version; otherwise use `Input` + `Static` fallback

### Validation behavior

Validation should be canonical and mostly flow through the same domain rules as `config.py`, rather than re-encoding a second truth table in the TUI.

Validation should:
- reject unsupported model kinds / dataset kinds / evaluation modes,
- reject non-numeric seed / threshold / folds / test size,
- reject invalid JSON in model params,
- require `dataset.path` when `dataset.kind == csv_tabular_binary`,
- require `folds >= 2` for stratified k-fold,
- avoid trying to train when validation fails.

Implementation rule:
- `DesignerDraft` holds UI-friendly strings / booleans / selections
- one normalization function converts that draft into a primitive config dict or directly into `TrainingConfig`
- validation is the result of that normalization attempt plus a few extra UI-level checks like JSON parsing and save-name hygiene

Validation output should be rendered in the preview / feedback area, not thrown as a raw exception.

### Dependent field rules

To keep the draft predictable, field dependencies must be explicit:
- changing `model.kind` resets `model.params` to that backend's default params
- changing `dataset.kind` to `sklearn_breast_cancer` clears `dataset.path` and de-emphasizes CSV-only fields
- changing `dataset.kind` to `csv_tabular_binary` requires `dataset.path`
- changing `evaluation.mode` to `holdout` keeps `split.test_size` and `split.stratify` active
- changing `evaluation.mode` to `stratified_k_fold` keeps `folds` active and de-emphasizes holdout-centric copy

The app does not need fancy dynamic hiding on day one, but it must keep the draft coherent.

### Save behavior

Saving should:
- normalize the current form into a `TrainingConfig`,
- write YAML into `configs/<slug>.yaml`,
- refresh the config browser,
- keep the designer draft loaded.

The save path should be editable via a draft config name field, but constrained to `configs/` in this first iteration.

If `configs/<slug>.yaml` already exists, v1 should **refuse to overwrite it by default** and surface a clear status message. Explicit overwrite behavior can come later once the UX for confirmation is designed.

If the operator has unsaved changes and tries to load defaults or clone another config, the app should refuse silently changing the draft. In v1, a simple status warning is acceptable if a full confirmation modal is too heavy, but the action must be blocked.

### Launch behavior

Launching should:
- validate the draft,
- save the draft to a real file in `configs/`,
- call the same training path the CLI uses,
- refresh dashboard summary + config list after success,
- surface the resulting run directory in the task-status panel.

Implementation should **not** shell out to `bc-mlops train` from the TUI callbacks. It should reuse Python-level config loading / training functions directly, to preserve testability and architectural sanity.

Preview semantics are precise: the preview pane always shows the **normalized YAML that will actually be saved and launched**.

The first implementation may run training synchronously and temporarily block the TUI while the run executes. That is an acceptable v1 tradeoff as long as the resulting status message clearly reports success or failure and the UI refreshes afterward.

## Architecture

To avoid making `tui.py` even more of a magnificent goblin cave, the run designer should introduce explicit state/helper layers in a **separate module now**, not later.

### New module

Create a dedicated module:
- `src/bc_mlops_showcase/designer.py`

It should hold the draft state model and the pure-ish helper functions that power the UI.

### New dataclasses / helpers in `designer.py`

Add:
- `DesignerDraft`
  - pure draft state for form fields
- `DesignerValidationResult`
  - `ok: bool`
  - `errors: list[str]`
  - `resolved_config: TrainingConfig | None`
- `DesignerActionResult`
  - `ok: bool`
  - `title: str`
  - `message: str`
  - `output: str`
  - `config_path: Path | None`
  - `run_dir: Path | None`
- helper functions:
  - `build_default_designer_draft() -> DesignerDraft`
  - `build_designer_draft_from_config(config: TrainingConfig, source_name: str | None = None) -> DesignerDraft`
  - `designer_draft_to_config(draft: DesignerDraft) -> TrainingConfig`
  - `validate_designer_draft(draft: DesignerDraft) -> DesignerValidationResult`
  - `render_designer_preview_text(draft: DesignerDraft) -> str`
  - `save_designer_draft(draft: DesignerDraft, config_root: Path) -> Path`
  - `launch_designer_run(draft: DesignerDraft, output_root: Path) -> DesignerActionResult`

These should stay mostly pure/testable and keep event handlers thin.

`launch_designer_run(...)` may internally call `save_designer_draft(...)` first, then `load_training_config(...)`, then `train_and_evaluate(...)` to stay aligned with the rest of the system.

### App state additions

Extend `MerlinDashboardApp` with:
- `mode: Literal["runs", "configs", "designer"]`
- `designer_draft: DesignerDraft`
- `designer_source_config_name: str | None`
- `designer_dirty: bool`
- widget IDs for designer inputs and controls

`tui.py` should remain responsible for:
- layout/widget wiring
- mode switching
- reading widget values into `DesignerDraft`
- painting preview/status output

It should **not** contain the main validation/save/launch business logic.

### Why a separate module now?

Because `tui.py` is already large and highly branched. The designer adds enough new state, validation, preview, and save/launch behavior that keeping it all in `tui.py` would turn the file from “large” into “sentient.”

## Testing strategy

Follow TDD and cover pure logic before app behavior.

### New pure-function tests

Add tests for:
- default draft values
- cloning a config into a draft
- converting a draft into `TrainingConfig`
- model-kind change resetting params defaults
- invalid JSON params rejection
- missing CSV path rejection
- invalid folds rejection
- YAML preview rendering
- saving a draft into `configs/`
- launching a run from a valid draft using the real training pipeline in a temp directory

### New interactive tests

Add tests for:
- opening designer mode from a button
- loading defaults into the form
- cloning selected config into the designer
- changing a few representative select/input values and seeing preview update after an explicit preview/validate action
- `Validate Draft` showing a validation result
- `Save Config` creating a config file
- `Launch Run` producing a run directory and refreshing the dashboard state

The interactive tests should continue to favor real widget events where practical, but most branchy validation should be proved in pure helper tests instead of brittle widget choreography.

## Files expected to change

### Code
- `src/bc_mlops_showcase/tui.py`
- `src/bc_mlops_showcase/designer.py`
- maybe `src/bc_mlops_showcase/cli.py` only if launch wiring needs explicit config roots passed more consistently
- maybe `src/bc_mlops_showcase/config.py` if a small serialization helper is needed for cleaner round-tripping

### Tests
- `tests/test_interactive_tui.py`
- likely a new pure-helper test module such as `tests/test_designer.py`
- maybe `tests/test_cli.py` if CLI-adjacent helpers change

### Docs
- `README.md`
- `docs/source/usage.rst`
- possibly generated API docs after Sphinx build if tracked artifacts change

## Open design choices

### Choice 1: save-before-launch vs temporary launch

Recommendation:
- **Launch Run always saves first.**
- Save to `configs/<slug>.yaml`, then train from that file.

Why:
- better reproducibility,
- clearer operator mental model,
- makes launched runs easier to trace back to a named config.

### Choice 2: free-form params UI vs field-per-backend UI

Recommendation:
- first iteration uses a **JSON params editor** for `model.params`
- plus finite-choice controls for the stable top-level fields

Why:
- supports all current backends with minimal bespoke UI work,
- avoids baking backend-specific assumptions into the first version,
- keeps the implementation smaller and easier to test.

### Choice 3: separate modal vs integrated lane

Recommendation:
- use an integrated `designer` mode/lane

Why:
- consistent with current runs/configs lane model,
- easier to test via stable widget IDs,
- less Textual complexity for a first pass.

### Choice 4: immediate validation vs explicit validation

Recommendation:
- use lightweight field syncing while editing
- use explicit `Preview YAML`, `Validate Draft`, `Save Config`, and `Launch Run` actions for full validation feedback

Why:
- less noisy in a terminal form,
- easier to test,
- avoids red-error confetti on every keystroke.

## Acceptance criteria

The feature is done when:
- the TUI can enter a designer mode,
- an operator can start from defaults or clone a config,
- the operator can edit the main config fields via widgets,
- the draft can be validated with readable feedback,
- the draft can be previewed as normalized YAML,
- the draft can be saved under `configs/`,
- the draft can launch a real training run,
- the config browser and run browser refresh after save/launch,
- tests, lint, and docs all pass.

## Implementation order

1. Add pure draft/validation/preview/save helpers in `designer.py` and tests.
2. Add designer mode state in the TUI and a minimal two-pane render.
3. Add designer widgets and wire controls.
4. Add save + launch actions.
5. Add docs.
6. Run independent review, fix, verify, ship.
