# Model registry refactor design

## Goal

Refactor the model registry so model metadata, default parameters, model-designer controls, and validation rules are declared once and consumed everywhere.

The immediate target is the model designer, but the design should also reduce drift across:

- `config.py`
- `designer.py`
- `model_designer.py`
- `tui.py`
- backend onboarding docs/tests

## Problem

The current code splits model knowledge across several brittle surfaces:

- `MODEL_SPECS` defines labels and compatibility metadata
- `DEFAULT_MODEL_PARAMS` separately defines defaults
- `model_designer.py` hard-codes per-backend field names, parsing, formatting, and validation branches
- `tui.py` hard-codes a fixed set of model-designer widgets and widget-to-draft mapping

That means each new backend risks touching multiple unrelated switch statements and form definitions. It works today, but only in the same way a pile of extension cords technically powers a house.

## Design principles

1. **Single declaration of model parameters**
   - each model backend declares its editable parameters once
   - defaults come from the same declaration
   - model-designer formatting/parsing rules come from the same declaration

2. **Registry-driven model designer**
   - the model designer draft should store generic parameter text values rather than per-backend dataclass fields
   - the TUI should render parameter widgets from registry metadata instead of hand-maintained field lists

3. **Canonical validation below the UI**
   - parsing and validation live in reusable config/model-designer helpers
   - the TUI only reflects those constraints early; it must not be the sole enforcer

4. **Extensible with modest ceremony**
   - adding a backend should mostly mean:
     1. register `ModelSpec`
     2. declare parameter schema
     3. implement trainer/inference loader if needed
     4. add tests

## Proposed model registry contract

Add a parameter-spec layer in `config.py`.

### `ModelParamSpec`

Each parameter spec declares:

- `name`: config key, e.g. `learning_rate`
- `label`: operator-facing label, e.g. `Learning rate`
- `kind`: one of `int`, `float`, `optional_int`, `int_list`
- `default`: canonical default value
- `placeholder`: optional TUI placeholder/help text
- `help_text`: optional operator-facing hint text
- `order`: stable form ordering

The initial `kind` set is intentionally small because the current supported models only need a few parsing strategies. The registry implementation should still keep parsing/formatting/validation centralized behind helper functions so future additions like booleans, enums, bounded numbers, or file paths do not reintroduce ad hoc branches in the TUI.

### `ModelSpec`

Extend `ModelSpec` to include:

- `parameter_specs: tuple[ModelParamSpec, ...]`

Derive defaults from `parameter_specs` instead of maintaining a separate `DEFAULT_MODEL_PARAMS` constant.

## Draft representation

Replace the current model-designer draft’s per-family fields with:

- `model_kind: str`
- `device: str`
- `param_values: dict[str, str]`
- `source_name: str | None`

This removes field explosion and lets the draft support future models without changing the dataclass definition.

## Conversion and validation flow

### Building a draft

`build_model_designer_draft_from_model_config()` should:

1. resolve the model spec
2. seed defaults from the registry
3. overlay the config’s actual params
4. format each param to a UI string using registry-driven formatting

### Converting a draft back to `ModelConfig`

`model_designer_draft_to_model_config()` should:

1. resolve the model spec
2. validate the device
3. iterate the registered parameter specs for the selected model
4. parse each draft string according to the spec kind
5. return normalized typed params

Unknown keys in `param_values` may be ignored when they do not belong to the selected model family. This keeps family switching and stale UI state from creating config garbage.

By contrast, persisted config loading should reject unknown `model.params` keys for the selected backend. YAML/config validation must stay strict so typos do not silently disappear and damage reproducibility.

## TUI refactor

Keep the lane and workflows the same, but change the form implementation:

1. mount a `#model-designer-params` container under the common kind/device controls
2. regenerate the parameter widgets from the selected model spec whenever the model family changes or the draft is refreshed
3. assign stable IDs like `model-designer-param-<param-name>`
4. read/write widget values by iterating the selected model spec instead of naming individual fields in code

Widget rebuild flow must be explicit:

1. capture current widget state into the current draft
2. change selected family or draft source
3. rebuild the parameter widget container
4. hydrate widgets from the new draft under change-event suppression
5. recompute preview/dirty state from the draft instead of widget-local state

This keeps the widget surface aligned with the registry and avoids hard-coded form drift.

## Scope for this refactor

### In scope

- add model parameter registry metadata
- derive defaults from the registry
- refactor `ModelDesignerDraft` to generic parameter storage
- make `model_designer.py` registry-driven
- make the model-designer TUI parameter widgets registry-driven
- update affected tests/docs

### Out of scope

- backend training dispatch refactor beyond what already exists
- artifact-loader registry
- task-aware metric/gate contracts
- plugin-discovered third-party model registries

## Tests to add or update

1. config/model registry tests
   - defaults derive from parameter specs
   - model specs expose expected parameter names for representative backends

2. model designer helper tests
   - building drafts from configs preserves values in `param_values`
   - parsing validates by declared parameter kind
   - every registered backend round-trips through the model designer helpers
   - applying a model draft still only mutates the model slice of a run draft

3. TUI tests
   - model designer renders only the active family’s registry-defined fields
   - switching family rebuilds the parameter widget set

4. backward-compatibility tests
   - representative existing YAML configs still resolve to the same normalized `ModelConfig`

## Risks

1. **TUI state sync bugs when rebuilding widgets**
   - mitigate with focused widget-refresh tests and by reusing the draft as the source of truth

2. **Future parameter kinds becoming ad hoc again**
   - mitigate by keeping parsing centralized behind spec-kind helpers instead of sprinkling custom parsing into the TUI

3. **Backward drift in docs/howto guidance**
   - mitigate by updating `docs/architecture.md` and `docs/source/howtos/add-backend.rst` after implementation

## Success criteria

The refactor is successful if adding a new model family with existing primitive parameter kinds no longer requires:

- editing the model-designer dataclass definition
- adding new hand-written TUI input widgets
- adding another `if/elif` parsing branch in `model_designer.py`

Instead, the model should appear by registering its metadata and parameter schema in one place.
