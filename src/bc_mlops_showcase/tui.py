"""Branded terminal dashboard for recent training runs and artifact health."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Literal

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

RunMode = Literal["runs", "configs"]
RunDetailMode = Literal["run", "artifacts", "actions", "compare", "help"]
HealthFilter = Literal[
    "all",
    "unhealthy_only",
    "card_missing",
    "model_missing",
    "metrics_missing",
    "registry_drift",
    "cv_only",
]

SORT_KEYS = (
    "f1",
    "accuracy",
    "roc_auc",
    "run_name",
    "issue_count",
    "timestamp",
    "model_kind",
    "evaluation_mode",
    "cv_f1_std",
)
DEFAULT_SORT_KEY = "f1"
HEALTH_FILTERS: tuple[HealthFilter, ...] = (
    "all",
    "unhealthy_only",
    "card_missing",
    "model_missing",
    "metrics_missing",
    "registry_drift",
    "cv_only",
)
ARTIFACT_KEYS = (
    "metrics",
    "metadata",
    "model_card",
    "config",
    "fold_metrics",
    "feature_importance",
)
MODE_OPTIONS: tuple[tuple[str, RunMode], ...] = (
    ("Runs", "runs"),
    ("Configs", "configs"),
)
SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("F1", "f1"),
    ("Accuracy", "accuracy"),
    ("ROC AUC", "roc_auc"),
    ("Run name", "run_name"),
    ("Issue count", "issue_count"),
    ("Timestamp", "timestamp"),
    ("Model kind", "model_kind"),
    ("Evaluation", "evaluation_mode"),
    ("CV F1 σ", "cv_f1_std"),
)
HEALTH_FILTER_OPTIONS: tuple[tuple[str, HealthFilter], ...] = (
    ("All runs", "all"),
    ("Unhealthy only", "unhealthy_only"),
    ("Missing cards", "card_missing"),
    ("Missing models", "model_missing"),
    ("Missing metrics", "metrics_missing"),
    ("Registry drift", "registry_drift"),
    ("CV only", "cv_only"),
)


@dataclass(slots=True)
class RunArtifactStatus:
    """File health for a single training run directory."""

    run_name: str
    model_kind: str
    model_status: str
    metrics_status: str
    card_status: str
    config_status: str
    feature_importance_status: str
    metadata_status: str


@dataclass(slots=True)
class DashboardSummary:
    """Derived dashboard state built from the registry and artifact tree."""

    runs: list[dict[str, object]]
    best_run: dict[str, object] | None
    artifact_statuses: list[RunArtifactStatus]
    orphan_run_dirs: list[str]
    missing_run_dirs: list[str]


@dataclass(slots=True)
class DashboardRunView:
    """Flattened run information for leaderboard and interactive widgets."""

    run_name: str
    model_kind: str
    accuracy: str
    f1: str
    roc_auc: str
    model_status: str
    metrics_status: str
    card_status: str
    issue_count: int
    timestamp: str
    evaluation_mode: str
    cv_f1_std: str


@dataclass(slots=True)
class ConfigView:
    """Small summary for one config file in the browser view."""

    name: str
    path: Path
    experiment_name: str
    model_kind: str
    dataset_kind: str
    target_column: str
    evaluation_mode: str
    tracking_experiment: str


@dataclass(slots=True)
class ActionResult:
    """Outcome from an in-TUI operator action."""

    ok: bool
    title: str
    message: str
    output: str


BRAND_TITLE = "MERLIN // ONCO-OPS COMMAND DECK"
BRAND_TAGLINE = "A slightly dramatic bridge view for breast-cancer MLOps runs."


class MerlinDashboardApp(App[None]):
    """Interactive Textual dashboard for exploring tracked runs."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #logo {
        height: auto;
        padding: 1 2;
        content-align: center middle;
        color: magenta;
        background: rgb(20, 7, 35);
        border: heavy rgb(167, 139, 250);
        margin: 0 1;
    }

    #run-filter {
        margin: 1 1 0 1;
    }

    #control-bar {
        height: auto;
        margin: 0 1;
    }

    .control-group {
        width: 1fr;
        height: auto;
        border: round rgb(59, 130, 246);
        padding: 0 1 1 1;
        margin-right: 1;
    }

    .control-group.-last {
        margin-right: 0;
    }

    .menu-select {
        margin-bottom: 1;
    }

    .toolbar-button {
        margin-right: 1;
        margin-bottom: 1;
        min-width: 12;
    }

    #main-grid {
        height: 1fr;
        margin: 1;
    }

    #sidebar {
        width: 1fr;
        height: 1fr;
        border: round rgb(59, 130, 246);
        padding: 0 1;
        margin-right: 1;
    }

    #inspector-pane {
        width: 1fr;
        height: 1fr;
    }

    #overview, #details-pane, #task-pane {
        border: round rgb(59, 130, 246);
        padding: 0 1;
    }

    #overview {
        height: 9;
        margin-bottom: 1;
    }

    #details-pane {
        height: 1fr;
        margin-bottom: 1;
    }

    #task-pane {
        height: 11;
    }

    .section-title {
        text-style: bold;
        color: cyan;
        margin-bottom: 1;
    }

    #run-list {
        height: 1fr;
    }

    #run-details, #task-status {
        height: 1fr;
        padding: 0 1;
    }

    #status-bar {
        height: auto;
        padding: 0 2 1 2;
        color: yellow;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload", "Reload"),
        Binding("slash", "focus_filter", "Filter"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("h", "toggle_health_filter", "Health"),
        Binding("tab", "switch_mode", "Mode", priority=True),
        Binding("enter", "open_detail", "Detail", priority=True),
        Binding("a", "show_actions", "Actions"),
        Binding("c", "compare_selected_run", "Compare"),
        Binding("question_mark", "toggle_help", "Help", priority=True),
        Binding("v", "validate_selected_run", "Validate"),
        Binding("m", "report_selected_run", "Model card"),
        Binding("p", "predict_selected_run", "Predict"),
        Binding("t", "retrain_selected_run", "Retrain"),
    ]

    def __init__(
        self,
        registry_path: Path,
        run_root: Path,
        config_root: Path = Path("configs"),
        sample_input_path: Path = Path("sample-inputs/sample.json"),
        gates_path: Path = Path("configs/quality_gates.yaml"),
    ) -> None:
        super().__init__()
        self.registry_path = registry_path
        self.run_root = run_root
        self.config_root = config_root
        self.sample_input_path = sample_input_path
        self.gates_path = gates_path
        self.summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
        self.config_views = load_config_views(config_root)
        self.sort_key = DEFAULT_SORT_KEY
        self.health_filter: HealthFilter = "all"
        self.mode: RunMode = "runs"
        self.detail_mode: RunDetailMode = "run"
        self.selected_run_name = _default_selected_run_name(self.summary)
        self.selected_config_name = self.config_views[0].name if self.config_views else None
        self.compare_anchor_name: str | None = None
        self.artifact_key = "metrics"
        self.last_action_result = ActionResult(
            ok=True,
            title="Idle",
            message="No action run yet.",
            output="Press a/v/m/p/t inside a run to operate from the command deck.",
        )
        self._is_refreshing_list = False
        self._is_syncing_controls = False
        self._prior_detail_mode: RunDetailMode = "run"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(render_merlin_logo(), id="logo")
        yield Input(placeholder="Filter runs or configs...", id="run-filter")
        with Horizontal(id="control-bar"):
            with Vertical(classes="control-group"):
                yield Static("Menus", classes="section-title")
                yield Select(
                    MODE_OPTIONS,
                    value=self.mode,
                    allow_blank=False,
                    id="mode-select",
                    classes="menu-select",
                )
                yield Select(
                    SORT_OPTIONS,
                    value=self.sort_key,
                    allow_blank=False,
                    id="sort-select",
                    classes="menu-select",
                )
                yield Select(
                    HEALTH_FILTER_OPTIONS,
                    value=self.health_filter,
                    allow_blank=False,
                    id="health-select",
                    classes="menu-select",
                )
            with Vertical(classes="control-group"):
                yield Static("Navigation", classes="section-title")
                with Horizontal():
                    yield Button("Reload", id="reload-button", classes="toolbar-button")
                    yield Button("Actions", id="actions-button", classes="toolbar-button")
                    yield Button("Compare", id="compare-button", classes="toolbar-button")
                    yield Button("Help", id="help-button", classes="toolbar-button")
            with Vertical(classes="control-group -last"):
                yield Static("Run Actions", classes="section-title")
                with Horizontal():
                    yield Button("Validate", id="validate-button", classes="toolbar-button")
                    yield Button("Report", id="report-button", classes="toolbar-button")
                    yield Button("Predict", id="predict-button", classes="toolbar-button")
                    yield Button("Retrain", id="retrain-button", classes="toolbar-button")
        with Horizontal(id="main-grid"):
            with Vertical(id="sidebar"):
                yield Static("Tracked Items", classes="section-title")
                yield ListView(id="run-list")
            with Vertical(id="inspector-pane"):
                yield Static("", id="overview")
                with Vertical(id="details-pane"):
                    yield Static("Run Details", classes="section-title")
                    yield Static("", id="run-details")
                with Vertical(id="task-pane"):
                    yield Static("Task Status", classes="section-title")
                    yield Static("", id="task-status")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_view(query="")
        self._refresh_controls()
        self.query_one("#run-list", ListView).focus()
        self._refresh_status("Ready. / filter • tab mode • enter detail • a actions • ? help.")
        self._refresh_task_status()

    def action_focus_filter(self) -> None:
        self.query_one("#run-filter", Input).focus()

    def action_switch_mode(self) -> None:
        self._set_mode("configs" if self.mode == "runs" else "runs")

    def action_cycle_sort(self) -> None:
        if self.mode != "runs":
            self._refresh_status("Sort applies to runs mode only.")
            self._refresh_controls()
            return
        current_index = SORT_KEYS.index(self.sort_key)
        self._set_sort_key(SORT_KEYS[(current_index + 1) % len(SORT_KEYS)])

    def action_toggle_health_filter(self) -> None:
        if self.mode != "runs":
            self._refresh_status("Health filters apply to runs mode only.")
            self._refresh_controls()
            return
        current_index = HEALTH_FILTERS.index(self.health_filter)
        self._set_health_filter(HEALTH_FILTERS[(current_index + 1) % len(HEALTH_FILTERS)])

    def action_reload(self) -> None:
        self.summary = load_dashboard_summary(
            registry_path=self.registry_path,
            run_root=self.run_root,
        )
        self.config_views = load_config_views(self.config_root)
        self._refresh_view(query=self._current_query())
        self._refresh_status("Reloaded registry and config browser.")

    def action_open_detail(self) -> None:
        if self.mode == "configs":
            self.detail_mode = "run"
            self._refresh_details()
            self._refresh_status("Config detail refreshed.")
            return
        if self.detail_mode != "artifacts":
            self.detail_mode = "artifacts"
            self.artifact_key = ARTIFACT_KEYS[0]
        else:
            artifact_index = ARTIFACT_KEYS.index(self.artifact_key)
            self.artifact_key = ARTIFACT_KEYS[(artifact_index + 1) % len(ARTIFACT_KEYS)]
        self._refresh_details()
        self._refresh_status(f"Artifact view: {self.artifact_key}.")

    def action_show_actions(self) -> None:
        if self.mode != "runs":
            self._refresh_status("Actions pane is only available in runs mode.")
            return
        self.detail_mode = "actions"
        self._refresh_details()
        self._refresh_status("Action catalog opened.")

    def action_toggle_help(self) -> None:
        if self.detail_mode == "help":
            self.detail_mode = self._prior_detail_mode
        else:
            self._prior_detail_mode = self.detail_mode
            self.detail_mode = "help"
        self._refresh_details()
        self._refresh_status("Keyboard help toggled.")

    def action_compare_selected_run(self) -> None:
        if self.mode != "runs":
            self._refresh_status("Compare mode only works in runs mode.")
            return
        if self.selected_run_name is None:
            self._refresh_status("No run selected to compare.")
            return
        if self.compare_anchor_name is None:
            self.compare_anchor_name = self.selected_run_name
            self._refresh_status(f"Pinned compare anchor: {self.compare_anchor_name}")
            return
        self.detail_mode = "compare"
        self._refresh_details()
        if self.compare_anchor_name == self.selected_run_name:
            self._refresh_status("Select another run to compare against the anchor.")
        else:
            self._refresh_status(
                f"Comparing {self.compare_anchor_name} vs {self.selected_run_name}."
            )

    def action_validate_selected_run(self) -> None:
        self._execute_selected_action("validate")

    def action_report_selected_run(self) -> None:
        self._execute_selected_action("report")

    def action_predict_selected_run(self) -> None:
        self._execute_selected_action("predict")

    def action_retrain_selected_run(self) -> None:
        self._execute_selected_action("retrain")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "run-filter":
            return
        self._refresh_view(query=event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "run-filter":
            return
        self._refresh_view(query=event.value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        del event
        if self._is_refreshing_list:
            return
        self._sync_selected_from_list()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        del event
        if self._is_refreshing_list:
            return
        self._sync_selected_from_list()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "reload-button":
            self.action_reload()
        elif button_id == "actions-button":
            self.action_show_actions()
        elif button_id == "compare-button":
            self.action_compare_selected_run()
        elif button_id == "help-button":
            self.action_toggle_help()
        elif button_id == "validate-button":
            self.action_validate_selected_run()
        elif button_id == "report-button":
            self.action_report_selected_run()
        elif button_id == "predict-button":
            self.action_predict_selected_run()
        elif button_id == "retrain-button":
            self.action_retrain_selected_run()

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._is_syncing_controls or event.select.id is None:
            return
        value = event.value
        if value == Select.BLANK:
            return
        if event.select.id == "mode-select":
            self._set_mode(value)
            return
        if event.select.id == "sort-select":
            self._set_sort_key(str(value))
            return
        if event.select.id == "health-select":
            self._set_health_filter(value)
            return

    def _execute_selected_action(self, action: str) -> None:
        if self.mode != "runs":
            self._refresh_status("Operator actions only run in runs mode.")
            return
        self.last_action_result = execute_run_action(
            self.summary,
            run_root=self.run_root,
            selected_run_name=self.selected_run_name,
            action=action,
            sample_input_path=self.sample_input_path,
            gates_path=self.gates_path,
        )
        self._refresh_task_status()
        self._refresh_status(self.last_action_result.message)
        self.summary = load_dashboard_summary(
            registry_path=self.registry_path,
            run_root=self.run_root,
        )
        self._refresh_view(query=self._current_query())

    def _current_query(self) -> str:
        return self.query_one("#run-filter", Input).value

    def _refresh_controls(self) -> None:
        self._is_syncing_controls = True
        self.query_one("#mode-select", Select).value = self.mode
        self.query_one("#sort-select", Select).value = self.sort_key
        self.query_one("#health-select", Select).value = self.health_filter
        self._is_syncing_controls = False

    def _set_mode(self, mode: RunMode) -> None:
        if mode == self.mode:
            self._refresh_controls()
            return
        self.mode = mode
        self.detail_mode = "run"
        self._refresh_view(query=self._current_query())
        self._refresh_status("Switched workspace lane.")

    def _set_sort_key(self, sort_key: str) -> None:
        if self.mode != "runs":
            self._refresh_status("Sort applies to runs mode only.")
            self._refresh_controls()
            return
        if sort_key == self.sort_key:
            self._refresh_controls()
            return
        self.sort_key = sort_key
        self.selected_run_name = None
        self._refresh_view(query=self._current_query())
        self._refresh_status(f"Sort set to {self.sort_key}.")

    def _set_health_filter(self, health_filter: HealthFilter) -> None:
        if self.mode != "runs":
            self._refresh_status("Health filters apply to runs mode only.")
            self._refresh_controls()
            return
        if health_filter == self.health_filter:
            self._refresh_controls()
            return
        self.health_filter = health_filter
        self.selected_run_name = None
        self._refresh_view(query=self._current_query())
        self._refresh_status(
            f"Health filter set to {_health_filter_label(self.health_filter)}."
        )

    def _current_visible_runs(self) -> list[DashboardRunView]:
        return select_run_views(
            self.summary,
            query=self._current_query(),
            sort_key=self.sort_key,
            unhealthy_only=self.health_filter == "unhealthy_only",
            health_filter=self.health_filter,
        )

    def _current_visible_configs(self) -> list[ConfigView]:
        return select_config_views(self.config_views, query=self._current_query())

    def _refresh_view(self, query: str) -> None:
        self._refresh_list(query=query)
        self._refresh_details()
        self._refresh_overview()
        self._refresh_task_status()
        self._refresh_controls()

    def _refresh_list(self, query: str) -> None:
        run_list = self.query_one("#run-list", ListView)
        self._is_refreshing_list = True
        run_list.clear()

        if self.mode == "runs":
            run_views = select_run_views(
                self.summary,
                query=query,
                sort_key=self.sort_key,
                unhealthy_only=self.health_filter == "unhealthy_only",
                health_filter=self.health_filter,
            )
            if not run_views:
                self.selected_run_name = None
                run_list.append(ListItem(Label("No runs match that filter."), name=""))
                self._is_refreshing_list = False
                return

            selected_name = self.selected_run_name
            visible_run_names = {run.run_name for run in run_views}
            if selected_name not in visible_run_names:
                selected_name = run_views[0].run_name

            selected_index = 0
            for index, run_view in enumerate(run_views):
                issue_marker = f" · issues={run_view.issue_count}" if run_view.issue_count else ""
                label = (
                    f"{run_view.run_name} [{run_view.model_kind}] "
                    f"F1={run_view.f1}{issue_marker}"
                )
                run_list.append(ListItem(Label(label), name=run_view.run_name))
                if run_view.run_name == selected_name:
                    selected_index = index

            run_list.index = selected_index
            self.selected_run_name = run_views[selected_index].run_name
            self._is_refreshing_list = False
            return

        config_views = select_config_views(self.config_views, query=query)
        if not config_views:
            self.selected_config_name = None
            run_list.append(ListItem(Label("No configs match that filter."), name=""))
            self._is_refreshing_list = False
            return

        selected_name = self.selected_config_name
        visible_config_names = {config.name for config in config_views}
        if selected_name not in visible_config_names:
            selected_name = config_views[0].name

        selected_index = 0
        for index, config_view in enumerate(config_views):
            label = (
                f"{config_view.name} [{config_view.model_kind}] "
                f"dataset={config_view.dataset_kind}"
            )
            run_list.append(ListItem(Label(label), name=config_view.name))
            if config_view.name == selected_name:
                selected_index = index

        run_list.index = selected_index
        self.selected_config_name = config_views[selected_index].name
        self._is_refreshing_list = False

    def _refresh_overview(self) -> None:
        overview = self.query_one("#overview", Static)
        if self.mode == "runs":
            overview.update(
                build_overview_text(
                    self.summary,
                    self._current_visible_runs(),
                    sort_key=self.sort_key,
                    unhealthy_only=self.health_filter == "unhealthy_only",
                    query=self._current_query(),
                    health_filter=self.health_filter,
                )
            )
            return
        overview.update(
            build_config_overview_text(
                self.config_views,
                self._current_visible_configs(),
                self._current_query(),
            )
        )

    def _sync_selected_from_list(self) -> None:
        run_list = self.query_one("#run-list", ListView)
        if run_list.index is None:
            return
        if not (0 <= run_list.index < len(run_list.children)):
            return
        item = run_list.children[run_list.index]
        if not getattr(item, "name", None):
            return
        if self.mode == "runs":
            self.selected_run_name = item.name
            if self.detail_mode == "compare" and self.compare_anchor_name == self.selected_run_name:
                self.detail_mode = "run"
        else:
            self.selected_config_name = item.name
        self._refresh_details()
        self._refresh_overview()

    def _refresh_details(self) -> None:
        details = self.query_one("#run-details", Static)
        if self.detail_mode == "help":
            details.update(build_help_text())
            return

        if self.mode == "configs":
            details.update(build_selected_config_text(self.config_views, self.selected_config_name))
            return

        visible_runs = self._current_visible_runs()
        if not visible_runs:
            details.update("No runs match that filter. Clear it or reload with r.")
            return

        if self.detail_mode == "artifacts":
            details.update(
                build_artifact_detail_text(
                    self.summary,
                    run_root=self.run_root,
                    selected_run_name=self.selected_run_name,
                    artifact_key=self.artifact_key,
                )
            )
            return

        if self.detail_mode == "actions":
            details.update(build_action_catalog_text(self.selected_run_name))
            return

        if self.detail_mode == "compare":
            details.update(
                build_compare_text(
                    self.summary,
                    self.compare_anchor_name,
                    self.selected_run_name,
                )
            )
            return

        details.update(build_run_detail_text(self.summary, self.selected_run_name))

    def _refresh_task_status(self) -> None:
        task_status = self.query_one("#task-status", Static)
        icon = "✅" if self.last_action_result.ok else "⚠️"
        task_status.update(
            "\n".join(
                [
                    f"{icon} {self.last_action_result.title}",
                    self.last_action_result.message,
                    "",
                    self.last_action_result.output,
                ]
            )
        )

    def _refresh_status(self, message: str) -> None:
        self.query_one("#status-bar", Static).update(f"Mode: {self.mode} • {message}")


def load_dashboard_summary(registry_path: Path, run_root: Path) -> DashboardSummary:
    """Load registry data and enrich it with artifact presence checks."""

    payload: dict[str, object] = {"runs": []}
    if registry_path.exists():
        payload = _safe_json_file(registry_path, default=payload)

    runs = _normalize_runs(payload.get("runs", []))
    best_run = _normalize_run(payload.get("best_run"))
    if best_run is None and runs:
        best_run = max(
            runs,
            key=lambda run: (_metric_value(run, "f1"), _metric_value(run, "roc_auc")),
        )

    artifact_statuses: list[RunArtifactStatus] = []
    enriched_runs: list[dict[str, object]] = []
    for run in runs:
        run_dir = run_root / str(run["run_name"])
        metadata = _load_run_metadata(run_dir)
        model_artifact = str(metadata.get("model_artifact", "model artifact"))
        enriched_run = dict(run)
        enriched_run["model_kind"] = enriched_run.get("model_kind") or metadata.get(
            "model_kind", "unknown"
        )
        for key in (
            "timestamp",
            "experiment_name",
            "train_rows",
            "test_rows",
            "evaluation_mode",
            "evaluation_folds",
            "cv_f1_mean",
            "cv_f1_std",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "dataset_kind",
            "dataset_path",
            "target_column",
            "mlflow_run_id",
            "mlflow_tracking_uri",
            "runtime_framework",
            "runtime_device",
            "model_artifact",
        ):
            if key not in enriched_run and metadata.get(key) is not None:
                enriched_run[key] = metadata[key]
        enriched_runs.append(enriched_run)
        artifact_statuses.append(
            RunArtifactStatus(
                run_name=str(enriched_run["run_name"]),
                model_kind=str(enriched_run["model_kind"]),
                model_status=_ok_or_missing(run_dir / model_artifact, model_artifact),
                metrics_status=_ok_or_missing(run_dir / "metrics.json", "metrics.json"),
                card_status=_ok_or_missing(run_dir / "MODEL_CARD.md", "MODEL_CARD"),
                config_status=_ok_or_missing(
                    run_dir / "config.resolved.yaml", "config.resolved.yaml"
                ),
                feature_importance_status=_ok_or_missing(
                    run_dir / "feature_importance.csv", "feature_importance.csv"
                ),
                metadata_status=_ok_or_missing(run_dir / "metadata.json", "metadata.json"),
            )
        )

    runs = enriched_runs
    if best_run is not None:
        best_run = next(
            (run for run in runs if run["run_name"] == best_run["run_name"]),
            best_run,
        )

    disk_run_dirs = (
        sorted(path.name for path in run_root.iterdir() if path.is_dir())
        if run_root.exists()
        else []
    )
    registry_run_names = sorted(str(run["run_name"]) for run in runs)
    registry_run_name_set = set(registry_run_names)
    disk_run_dir_set = set(disk_run_dirs)

    return DashboardSummary(
        runs=runs,
        best_run=best_run,
        artifact_statuses=artifact_statuses,
        orphan_run_dirs=sorted(disk_run_dir_set - registry_run_name_set),
        missing_run_dirs=sorted(registry_run_name_set - disk_run_dir_set),
    )


def load_config_views(config_root: Path) -> list[ConfigView]:
    """Load browsable config summaries from a config directory."""

    from .config import load_training_config

    if not config_root.exists():
        return []

    views: list[ConfigView] = []
    for path in sorted(config_root.glob("*.yaml")):
        try:
            config = load_training_config(path)
        except Exception:
            continue
        evaluation_mode = config.evaluation.mode
        if evaluation_mode == "stratified_k_fold":
            evaluation_mode = f"{evaluation_mode} ({config.evaluation.folds} folds)"
        views.append(
            ConfigView(
                name=path.stem,
                path=path,
                experiment_name=config.experiment_name,
                model_kind=config.model.kind,
                dataset_kind=config.dataset.kind,
                target_column=config.dataset.target_column,
                evaluation_mode=evaluation_mode,
                tracking_experiment=config.tracking.experiment_name,
            )
        )
    return views


def select_config_views(config_views: list[ConfigView], query: str) -> list[ConfigView]:
    """Filter config views for the config browser mode."""

    needle = query.strip().lower()
    if not needle:
        return list(config_views)
    return [
        config_view
        for config_view in config_views
        if needle in config_view.name.lower()
        or needle in config_view.model_kind.lower()
        or needle in config_view.dataset_kind.lower()
        or needle in config_view.experiment_name.lower()
    ]


def render_merlin_logo() -> str:
    """Return the reusable Merlin-themed text logo for terminal views."""

    return (
        "        /\\\n"
        "   /\\  /  \\\n"
        "  /__\\/ /\\ \\\n"
        " /\\  / ____  \\\n"
        "/__\\/_/    \\_\\   MERLIN\n"
        "   ✦   ONCO-OPS Command Deck   ✦"
    )


def _default_selected_run_name(summary: DashboardSummary) -> str | None:
    if summary.best_run is not None:
        return str(summary.best_run["run_name"])
    if summary.runs:
        return str(summary.runs[0]["run_name"])
    return None


def _artifact_status_by_run(summary: DashboardSummary) -> dict[str, RunArtifactStatus]:
    return {status.run_name: status for status in summary.artifact_statuses}


def _artifact_issue_count(status: RunArtifactStatus) -> int:
    return sum(
        not value.startswith("OK:")
        for value in (status.model_status, status.metrics_status, status.card_status)
    )


def build_run_views(summary: DashboardSummary) -> list[DashboardRunView]:
    """Join registry rows with artifact health for rendering and selection."""

    statuses = _artifact_status_by_run(summary)
    views: list[DashboardRunView] = []
    for run in summary.runs:
        run_name = str(run["run_name"])
        status = statuses.get(
            run_name,
            RunArtifactStatus(
                run_name=run_name,
                model_kind=str(run.get("model_kind", "unknown")),
                model_status="model artifact missing",
                metrics_status="metrics.json missing",
                card_status="MODEL_CARD missing",
                config_status="config.resolved.yaml missing",
                feature_importance_status="feature_importance.csv missing",
                metadata_status="metadata.json missing",
            ),
        )
        issue_count = _artifact_issue_count(status)
        views.append(
            DashboardRunView(
                run_name=run_name,
                model_kind=str(run.get("model_kind", "unknown")),
                accuracy=_format_metric(run, "accuracy"),
                f1=_format_metric(run, "f1"),
                roc_auc=_format_metric(run, "roc_auc"),
                model_status=status.model_status,
                metrics_status=status.metrics_status,
                card_status=status.card_status,
                issue_count=issue_count,
                timestamp=str(run.get("timestamp", "n/a")),
                evaluation_mode=_format_evaluation_strategy(run),
                cv_f1_std=_format_summary_metric(run, "cv_f1_std"),
            )
        )
    return views


def select_run_views(
    summary: DashboardSummary,
    *,
    query: str,
    sort_key: str,
    unhealthy_only: bool,
    health_filter: HealthFilter | None = None,
) -> list[DashboardRunView]:
    """Filter and sort run views for static and interactive presentations."""

    run_views = build_run_views(summary)
    needle = query.strip().lower()
    if needle:
        run_views = [
            run_view
            for run_view in run_views
            if needle in run_view.run_name.lower() or needle in run_view.model_kind.lower()
        ]

    effective_filter: HealthFilter = health_filter or (
        "unhealthy_only" if unhealthy_only else "all"
    )
    run_views = _apply_health_filter(summary, run_views, effective_filter)

    return sorted(
        run_views,
        key=lambda run_view: _sort_value(run_view, sort_key),
        reverse=sort_key != "run_name",
    )


def build_overview_text(
    summary: DashboardSummary,
    visible_runs: list[DashboardRunView],
    *,
    sort_key: str,
    unhealthy_only: bool,
    query: str,
    health_filter: HealthFilter | None = None,
) -> str:
    """Build summary text for the interactive inspector pane."""

    champion_name = str(summary.best_run["run_name"]) if summary.best_run is not None else "n/a"
    unhealthy_count = sum(status_has_issues(status) for status in summary.artifact_statuses)
    effective_filter = health_filter or ("unhealthy_only" if unhealthy_only else "all")
    search_text = query.strip() or "—"
    lines = [
        "Deck Overview",
        f"Tracked runs: {len(summary.runs)}",
        f"Visible runs: {len(visible_runs)}",
        f"Champion: {champion_name}",
        f"Runs with artifact issues: {unhealthy_count}",
        f"Orphan run dirs: {len(summary.orphan_run_dirs)}",
        f"Registry entries without run dirs: {len(summary.missing_run_dirs)}",
        f"Sort: {sort_key}",
        f"Health filter: {_health_filter_label(effective_filter)}",
        f"Search: {search_text}",
    ]
    if summary.orphan_run_dirs:
        lines.append(f"Orphans: {', '.join(summary.orphan_run_dirs)}")
    if summary.missing_run_dirs:
        lines.append(f"Missing dirs: {', '.join(summary.missing_run_dirs)}")
    return "\n".join(lines)


def build_config_overview_text(
    config_views: list[ConfigView], visible_configs: list[ConfigView], query: str
) -> str:
    """Build summary text for config-browser mode."""

    search_text = query.strip() or "—"
    return "\n".join(
        [
            "Config Browser",
            f"Tracked configs: {len(config_views)}",
            f"Visible configs: {len(visible_configs)}",
            f"Search: {search_text}",
            "Use tab to switch back to runs.",
            "Use t from a run to retrain from config.resolved.yaml.",
        ]
    )


def build_run_detail_text(summary: DashboardSummary, selected_run_name: str | None) -> str:
    """Build the right-hand detail pane text for one selected run."""

    if not summary.runs:
        return "No tracked runs yet. Train a model to populate the command deck."
    if selected_run_name is None:
        return "No run selected. Use the filter, health toggle, or arrow keys to pick a run."

    run_by_name = {str(run["run_name"]): run for run in summary.runs}
    status_by_name = _artifact_status_by_run(summary)
    run = run_by_name.get(selected_run_name)
    status = status_by_name.get(selected_run_name)
    if run is None or status is None:
        return "Selected run is no longer available. Reload the dashboard with r."

    champion_marker = (
        "Yes" if summary.best_run and summary.best_run["run_name"] == selected_run_name else "No"
    )
    delta_line = _build_delta_line(summary.best_run, run)
    issue_summary = f"Artifact issues: {_artifact_issue_count(status)}"
    timestamp = str(run.get("timestamp", "n/a"))
    train_rows = run.get("train_rows", "n/a")
    test_rows = run.get("test_rows", "n/a")
    runtime_framework = str(run.get("runtime_framework", "unknown"))
    runtime_device = str(run.get("runtime_device", "unknown"))
    evaluation_strategy = _format_evaluation_strategy(run)
    mlflow_run_id = str(run.get("mlflow_run_id", "n/a"))
    mlflow_tracking_uri = str(run.get("mlflow_tracking_uri", "n/a"))
    dataset_kind = str(run.get("dataset_kind", "unknown"))
    target_column = str(run.get("target_column", "n/a"))
    model_artifact = str(run.get("model_artifact", "unknown"))
    dataset_path = str(run.get("dataset_path", "n/a"))
    cv_f1_summary = _format_cv_summary_line(run, metric_key="f1", label="CV F1")
    cv_roc_auc_summary = _format_cv_summary_line(run, metric_key="roc_auc", label="CV ROC AUC")
    operator_actions = _operator_action_lines(selected_run_name)
    return "\n".join(
        [
            f"Run: {selected_run_name}",
            f"Champion: {champion_marker}",
            f"Model kind: {run.get('model_kind', 'unknown')}",
            f"Accuracy: {_format_metric(run, 'accuracy')}",
            f"F1: {_format_metric(run, 'f1')}",
            f"ROC AUC: {_format_metric(run, 'roc_auc')}",
            delta_line,
            issue_summary,
            "",
            "Run dossier:",
            f"- Timestamp: {timestamp}",
            f"- Rows: train={train_rows} test={test_rows}",
            f"- Evaluation: {evaluation_strategy}",
            f"- {cv_f1_summary}",
            f"- {cv_roc_auc_summary}",
            f"- Runtime: {runtime_framework} on {runtime_device}",
            f"- MLflow run id: {mlflow_run_id}",
            f"- Tracking URI: {mlflow_tracking_uri}",
            f"- Dataset: {dataset_kind}",
            f"- Dataset path: {dataset_path}",
            f"- Target column: {target_column}",
            f"- Model artifact: {model_artifact}",
            "",
            "Artifact health:",
            f"- {status.metadata_status}",
            f"- {status.model_status}",
            f"- {status.metrics_status}",
            f"- {status.card_status}",
            f"- {status.config_status}",
            f"- {status.feature_importance_status}",
            "",
            "Operator actions:",
            f"- {operator_actions[0]}",
            f"- {operator_actions[1]}",
        ]
    )


def build_config_detail_text(config_view: ConfigView) -> str:
    """Build a detailed description for a selected config."""

    return "\n".join(
        [
            f"Config: {config_view.name}",
            f"Path: {config_view.path}",
            f"Experiment: {config_view.experiment_name}",
            f"Model kind: {config_view.model_kind}",
            f"Dataset: {config_view.dataset_kind}",
            f"Target column: {config_view.target_column}",
            f"Evaluation: {config_view.evaluation_mode}",
            f"Tracking experiment: {config_view.tracking_experiment}",
            "",
            "Operator move:",
            f"- uv run bc-mlops train --config {config_view.path} --output-dir artifacts/runs",
        ]
    )


def build_selected_config_text(
    config_views: list[ConfigView], selected_config_name: str | None
) -> str:
    """Render selected config detail or an empty-state message."""

    if not config_views:
        return "No configs found. Add YAML files under configs/ to populate the browser."
    if selected_config_name is None:
        return "No config selected. Use the list to choose a config."
    config_by_name = {config_view.name: config_view for config_view in config_views}
    config_view = config_by_name.get(selected_config_name)
    if config_view is None:
        return "Selected config is no longer available. Reload with r."
    return build_config_detail_text(config_view)


def build_artifact_detail_text(
    summary: DashboardSummary,
    *,
    run_root: Path,
    selected_run_name: str | None,
    artifact_key: str,
) -> str:
    """Return raw run-file content for the artifact inspector view."""

    if selected_run_name is None:
        return "No run selected. Choose a run before opening artifact drill-down."
    artifact_path = _artifact_path_for_key(summary, run_root, selected_run_name, artifact_key)
    if artifact_path is None:
        return f"Artifact '{artifact_key}' is not available for this run."
    if not artifact_path.exists():
        return f"Artifact missing: {artifact_path.name}"
    try:
        content = artifact_path.read_text()
    except OSError as exc:
        return f"Could not read artifact {artifact_path.name}: {exc}"
    preview = content.strip() or "(empty file)"
    return "\n".join(
        [
            f"Artifact view: {artifact_key}",
            f"Path: {artifact_path}",
            "",
            preview,
        ]
    )


def build_compare_text(
    summary: DashboardSummary,
    left_run_name: str | None,
    right_run_name: str | None,
) -> str:
    """Build a run-vs-run comparison summary."""

    if left_run_name is None:
        return "Compare Runs\nPick a run and press c to set the compare anchor."
    if right_run_name is None or right_run_name == left_run_name:
        return (
            "Compare Runs\n"
            f"Anchor: {left_run_name}\n"
            "Select another run and press c again to compare."
        )

    run_by_name = {str(run["run_name"]): run for run in summary.runs}
    left = run_by_name.get(left_run_name)
    right = run_by_name.get(right_run_name)
    if left is None or right is None:
        return "Compare Runs\nOne of the selected runs is no longer present. Reload with r."

    left_status = _artifact_status_by_run(summary).get(left_run_name)
    right_status = _artifact_status_by_run(summary).get(right_run_name)
    left_issues = _artifact_issue_count(left_status) if left_status else 0
    right_issues = _artifact_issue_count(right_status) if right_status else 0
    return "\n".join(
        [
            "Compare Runs",
            f"Left: {left_run_name}",
            f"Right: {right_run_name}",
            "",
            f"ΔF1: {_metric_delta_text(right, left, 'f1')}",
            f"ΔROC AUC: {_metric_delta_text(right, left, 'roc_auc')}",
            f"ΔAccuracy: {_metric_delta_text(right, left, 'accuracy')}",
            f"Issue delta: {right_issues - left_issues:+d}",
            f"Left evaluation: {_format_evaluation_strategy(left)}",
            f"Right evaluation: {_format_evaluation_strategy(right)}",
            f"Left CV F1 σ: {_format_summary_metric(left, 'cv_f1_std')}",
            f"Right CV F1 σ: {_format_summary_metric(right, 'cv_f1_std')}",
        ]
    )


def build_action_catalog_text(selected_run_name: str | None) -> str:
    """Render the in-TUI action palette/help panel."""

    run_name = selected_run_name or "<selected-run>"
    return "\n".join(
        [
            "Action Catalog",
            f"Selected run: {run_name}",
            "",
            "Hotkeys:",
            "- v → validate selected run against configs/quality_gates.yaml",
            "- m → generate MODEL_CARD.md for the selected run",
            "- p → predict from sample-inputs/sample.json using the selected model",
            "- t → retrain from config.resolved.yaml for the selected run",
            "- enter → open/cycle artifact drill-down",
            "- c → pin compare anchor / compare current run",
        ]
    )


def build_help_text() -> str:
    """Render keyboard help for the interactive command deck."""

    return "\n".join(
        [
            "Keyboard Help",
            "- / focus filter",
            "- s cycle sort",
            "- h cycle triage filter",
            "- tab switch runs/configs workspace lanes",
            "- enter open artifact drill-down / cycle artifact file",
            "- a open action catalog",
            "- c compare selected run",
            "- v validate selected run",
            "- m generate model card",
            "- p run prediction",
            "- t retrain from resolved config",
            "- r reload registry and configs",
            "- q quit",
        ]
    )


def execute_run_action(
    summary: DashboardSummary,
    *,
    run_root: Path,
    selected_run_name: str | None,
    action: str,
    sample_input_path: Path = Path("sample-inputs/sample.json"),
    gates_path: Path = Path("configs/quality_gates.yaml"),
) -> ActionResult:
    """Run one operator workflow for the selected run."""

    if selected_run_name is None:
        return ActionResult(False, "No selection", "Pick a run before launching actions.", "")

    run_dir = run_root / selected_run_name
    if not run_dir.exists():
        return ActionResult(False, "Run missing", f"Run directory not found: {run_dir}", "")

    try:
        if action == "validate":
            from .validation import validate_metrics

            metrics_path = run_dir / "metrics.json"
            result = validate_metrics(metrics_path, gates_path)
            passed = bool(result.get("passed"))
            return ActionResult(
                passed,
                "Validation",
                f"Validation {'passed' if passed else 'failed'} for {selected_run_name}.",
                json.dumps(result, indent=2),
            )

        if action == "report":
            from .reporting import build_model_card

            output_path = run_dir / "MODEL_CARD.md"
            destination = build_model_card(run_dir, output_path)
            return ActionResult(
                True,
                "Model card",
                f"Generated model card for {selected_run_name}.",
                str(destination),
            )

        if action == "predict":
            from .inference import predict_records

            model_path = _artifact_path_for_key(summary, run_root, selected_run_name, "model")
            if model_path is None:
                return ActionResult(
                    False,
                    "Prediction",
                    f"Could not locate model artifact for {selected_run_name}.",
                    "",
                )
            prediction = predict_records(model_path=model_path, input_path=sample_input_path)
            return ActionResult(
                True,
                "Prediction",
                f"Scored sample input with {selected_run_name}.",
                json.dumps(prediction, indent=2),
            )

        if action == "retrain":
            from .config import load_training_config
            from .pipeline import train_and_evaluate

            config_path = run_dir / "config.resolved.yaml"
            config = load_training_config(config_path)
            result = train_and_evaluate(config=config, output_root=run_root)
            return ActionResult(
                True,
                "Retrain",
                f"Retraining launched from {config_path.name}.",
                json.dumps(result.summary(), indent=2),
            )

        return ActionResult(False, "Unknown action", f"Unsupported action: {action}", "")
    except Exception as exc:  # pragma: no cover - exercised by interactive flows
        return ActionResult(
            False,
            action.title(),
            f"{action} failed for {selected_run_name}.",
            str(exc),
        )


def launch_dashboard_app(
    registry_path: Path,
    run_root: Path,
    config_root: Path = Path("configs"),
    sample_input_path: Path = Path("sample-inputs/sample.json"),
    gates_path: Path = Path("configs/quality_gates.yaml"),
) -> int:
    """Launch the interactive Textual app."""

    app = MerlinDashboardApp(
        registry_path=registry_path,
        run_root=run_root,
        config_root=config_root,
        sample_input_path=sample_input_path,
        gates_path=gates_path,
    )
    app.run()
    return 0


def _load_run_metadata(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}

    payload = _safe_json_file(metadata_path, default={})
    fold_metrics = _safe_json_file(run_dir / "fold_metrics.json", default={})
    fold_summary = (
        fold_metrics.get("summary", {}) if isinstance(fold_metrics.get("summary", {}), dict) else {}
    )
    f1_summary = fold_summary.get("f1", {}) if isinstance(fold_summary.get("f1", {}), dict) else {}
    roc_auc_summary = (
        fold_summary.get("roc_auc", {}) if isinstance(fold_summary.get("roc_auc", {}), dict) else {}
    )
    model = payload.get("model", {}) if isinstance(payload.get("model", {}), dict) else {}
    runtime = model.get("runtime", {}) if isinstance(model.get("runtime", {}), dict) else {}
    dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset", {}), dict) else {}
    evaluation = (
        payload.get("evaluation", {})
        if isinstance(payload.get("evaluation", {}), dict)
        else {}
    )
    mlflow = payload.get("mlflow", {}) if isinstance(payload.get("mlflow", {}), dict) else {}
    return {
        "model_artifact": str(model.get("artifact", "model artifact")),
        "model_kind": str(model.get("kind", "unknown")),
        "timestamp": payload.get("timestamp"),
        "experiment_name": payload.get("experiment_name"),
        "train_rows": payload.get("train_rows"),
        "test_rows": payload.get("test_rows"),
        "evaluation_mode": evaluation.get("mode"),
        "evaluation_folds": evaluation.get("folds"),
        "cv_f1_mean": f1_summary.get("mean"),
        "cv_f1_std": f1_summary.get("std"),
        "cv_roc_auc_mean": roc_auc_summary.get("mean"),
        "cv_roc_auc_std": roc_auc_summary.get("std"),
        "dataset_kind": dataset.get("kind"),
        "dataset_path": dataset.get("path"),
        "target_column": dataset.get("target_column"),
        "mlflow_run_id": mlflow.get("run_id"),
        "mlflow_tracking_uri": mlflow.get("tracking_uri"),
        "runtime_framework": runtime.get("framework"),
        "runtime_device": runtime.get("device"),
    }


def _safe_json_file(path: Path, default: dict[str, object]) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default

    return payload if isinstance(payload, dict) else default


def _normalize_run(candidate: object) -> dict[str, object] | None:
    if not isinstance(candidate, dict):
        return None
    if "run_name" not in candidate:
        return None
    return dict(candidate)


def _normalize_runs(candidates: object) -> list[dict[str, object]]:
    if not isinstance(candidates, list):
        return []

    runs: list[dict[str, object]] = []
    for candidate in candidates:
        run = _normalize_run(candidate)
        if run is not None:
            runs.append(run)
    return runs


def _metric_value(run: dict[str, object], key: str, default: float = -1.0) -> float:
    value = run.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sort_value(run_view: DashboardRunView, sort_key: str) -> float | str:
    if sort_key in {"run_name", "timestamp", "model_kind", "evaluation_mode"}:
        return getattr(run_view, sort_key)
    value = getattr(run_view, sort_key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def status_has_issues(status: RunArtifactStatus) -> bool:
    """Return whether any artifact is missing for a run."""

    return _artifact_issue_count(status) > 0


def _build_delta_line(best_run: dict[str, object] | None, current_run: dict[str, object]) -> str:
    if best_run is None:
        return "Delta vs champion (F1): n/a"

    return f"Delta vs champion (F1): {_format_delta_vs_champion(best_run, current_run)}"


def _format_evaluation_strategy(run: dict[str, object]) -> str:
    mode = run.get("evaluation_mode")
    if not mode:
        return "n/a"

    try:
        folds = int(run.get("evaluation_folds"))
    except (TypeError, ValueError):
        return str(mode)

    if str(mode) != "stratified_k_fold":
        return str(mode)
    return f"{mode} ({folds} folds)"


def _format_delta_vs_champion(
    best_run: dict[str, object] | None, current_run: dict[str, object]
) -> str:
    if best_run is None:
        return "n/a"

    delta = _metric_value(current_run, "f1") - _metric_value(best_run, "f1")
    return f"{delta:+.4f}"


def _metric_delta_text(
    current_run: dict[str, object], baseline_run: dict[str, object], key: str
) -> str:
    delta = _metric_value(current_run, key) - _metric_value(baseline_run, key)
    return f"{delta:+.4f}"


def _format_metric(run: dict[str, object], key: str) -> str:
    value = run.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_summary_metric(run: dict[str, object], key: str) -> str:
    value = run.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_cv_summary_line(run: dict[str, object], *, metric_key: str, label: str) -> str:
    mean_key = f"cv_{metric_key}_mean"
    std_key = f"cv_{metric_key}_std"
    mean_text = _format_summary_metric(run, mean_key)
    std_text = _format_summary_metric(run, std_key)
    if mean_text == "n/a" and std_text == "n/a":
        return f"{label}: n/a"
    return f"{label}: mean={mean_text} std={std_text}"


def _ok_or_missing(path: Path, label: str) -> str:
    return f"OK: {label}" if path.exists() else f"{label} missing"


def _health_filter_label(health_filter: HealthFilter) -> str:
    return {
        "all": "all runs",
        "unhealthy_only": "unhealthy only",
        "card_missing": "missing model cards",
        "model_missing": "missing model artifacts",
        "metrics_missing": "missing metrics",
        "registry_drift": "registry drift",
        "cv_only": "cross-validation only",
    }[health_filter]


def _apply_health_filter(
    summary: DashboardSummary, run_views: list[DashboardRunView], health_filter: HealthFilter
) -> list[DashboardRunView]:
    if health_filter == "all":
        return run_views
    if health_filter == "unhealthy_only":
        return [run_view for run_view in run_views if run_view.issue_count > 0]
    if health_filter == "card_missing":
        return [run_view for run_view in run_views if "missing" in run_view.card_status]
    if health_filter == "model_missing":
        return [run_view for run_view in run_views if "missing" in run_view.model_status]
    if health_filter == "metrics_missing":
        return [run_view for run_view in run_views if "missing" in run_view.metrics_status]
    if health_filter == "registry_drift":
        drift_names = set(summary.orphan_run_dirs) | set(summary.missing_run_dirs)
        return [run_view for run_view in run_views if run_view.run_name in drift_names]
    if health_filter == "cv_only":
        return [
            run_view
            for run_view in run_views
            if run_view.evaluation_mode.startswith("stratified_k_fold")
        ]
    return run_views


def _artifact_path_for_key(
    summary: DashboardSummary, run_root: Path, run_name: str, artifact_key: str
) -> Path | None:
    run_dir = run_root / run_name
    if artifact_key == "metrics":
        return run_dir / "metrics.json"
    if artifact_key == "metadata":
        return run_dir / "metadata.json"
    if artifact_key == "model_card":
        return run_dir / "MODEL_CARD.md"
    if artifact_key == "config":
        return run_dir / "config.resolved.yaml"
    if artifact_key == "fold_metrics":
        return run_dir / "fold_metrics.json"
    if artifact_key == "feature_importance":
        return run_dir / "feature_importance.csv"
    if artifact_key == "model":
        run_by_name = {str(run["run_name"]): run for run in summary.runs}
        run = run_by_name.get(run_name, {})
        artifact_name = str(run.get("model_artifact", "model artifact"))
        return run_dir / artifact_name
    return None


def render_dashboard_text(
    registry_path: Path, run_root: Path, width: int = 110, *, color: bool = False
) -> str:
    """Render the dashboard to plain text for CLI output and tests."""

    summary = load_dashboard_summary(registry_path=registry_path, run_root=run_root)
    stream = StringIO()
    console = Console(
        file=stream,
        record=True,
        width=width,
        force_terminal=color,
        color_system="truecolor" if color else None,
    )
    console.print(_build_dashboard(summary, registry_path=registry_path, run_root=run_root))
    return stream.getvalue()


def _build_dashboard(summary: DashboardSummary, registry_path: Path, run_root: Path) -> Group:
    return Group(
        _brand_panel(),
        _summary_panel(summary),
        _leaderboard_panel(summary),
        _comparison_panel(summary),
        _artifact_health_panel(summary),
        _registry_drift_panel(summary),
        _operator_hints_panel(summary, registry_path=registry_path, run_root=run_root),
    )


def _brand_panel() -> Panel:
    title = Text(BRAND_TITLE, style="bold magenta")
    subtitle = Text(BRAND_TAGLINE, style="italic cyan")
    logo = Text(render_merlin_logo(), style="bright_magenta")
    banner = Group(logo, Text(""), title, Text(""), subtitle)
    return Panel(banner, border_style="bright_magenta", title="Command Bridge")


def _summary_panel(summary: DashboardSummary) -> Panel:
    if not summary.runs or summary.best_run is None:
        return Panel(
            Group(
                Text("No tracked runs yet.", style="bold yellow"),
                Text("Train with: bc-mlops train --config configs/train.yaml"),
            ),
            border_style="yellow",
            title="Champion Run",
        )

    best_run = summary.best_run
    lines = Group(
        Text(f"Champion: {best_run['run_name']}", style="bold green"),
        Text(f"Model kind: {best_run.get('model_kind', 'unknown')}"),
        Text(f"Evaluation: {_format_evaluation_strategy(best_run)}"),
        Text(f"Accuracy: {_format_metric(best_run, 'accuracy')}"),
        Text(f"F1: {_format_metric(best_run, 'f1')}"),
        Text(f"ROC AUC: {_format_metric(best_run, 'roc_auc')}"),
        Text(f"Tracked runs: {len(summary.runs)}"),
    )
    return Panel(lines, border_style="green", title="Champion Run")


def _leaderboard_panel(summary: DashboardSummary) -> Panel:
    table = Table(expand=True)
    table.add_column("Run")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ROC AUC", justify="right")
    table.add_column("Issues", justify="right")

    for run_view in select_run_views(
        summary,
        query="",
        sort_key=DEFAULT_SORT_KEY,
        unhealthy_only=False,
        health_filter="all",
    ):
        table.add_row(
            run_view.run_name,
            run_view.model_kind,
            run_view.accuracy,
            run_view.f1,
            run_view.roc_auc,
            str(run_view.issue_count),
        )

    if not summary.runs:
        table.add_row("—", "—", "—", "—", "—", "—")

    return Panel(table, border_style="blue", title="Leaderboard")


def _comparison_panel(summary: DashboardSummary) -> Panel:
    table = Table(expand=True)
    table.add_column("Rank", justify="right", no_wrap=True)
    table.add_column("Run")
    table.add_column("Model", no_wrap=True)
    table.add_column("F1", justify="right", no_wrap=True)
    table.add_column("ΔF1 vs champ", justify="right", no_wrap=True)
    table.add_column("F1 σ", justify="right", no_wrap=True)

    run_views = select_run_views(
        summary,
        query="",
        sort_key=DEFAULT_SORT_KEY,
        unhealthy_only=False,
        health_filter="all",
    )
    run_by_name = {str(run["run_name"]): run for run in summary.runs}

    for index, run_view in enumerate(run_views, start=1):
        run = run_by_name.get(run_view.run_name)
        if run is None:
            continue
        table.add_row(
            str(index),
            run_view.run_name,
            run_view.model_kind,
            run_view.f1,
            _format_delta_vs_champion(summary.best_run, run),
            _format_summary_metric(run, "cv_f1_std"),
        )

    if not run_views:
        table.add_row("—", "—", "—", "—", "—", "—")

    return Panel(table, border_style="magenta", title="Compare View")


def _artifact_health_panel(summary: DashboardSummary) -> Panel:
    table = Table(expand=True)
    table.add_column("Run")
    table.add_column("Model kind", no_wrap=True)
    table.add_column("Model artifact", no_wrap=True)
    table.add_column("Metrics", no_wrap=True)
    table.add_column("Model card", no_wrap=True)

    for status in summary.artifact_statuses:
        table.add_row(
            status.run_name,
            status.model_kind,
            status.model_status,
            status.metrics_status,
            status.card_status,
        )

    if not summary.artifact_statuses:
        table.add_row("—", "—", "—", "—", "—")

    return Panel(table, border_style="red", title="Artifact Health")


def _registry_drift_panel(summary: DashboardSummary) -> Panel:
    orphan_runs = ", ".join(summary.orphan_run_dirs) if summary.orphan_run_dirs else "none"
    missing_runs = ", ".join(summary.missing_run_dirs) if summary.missing_run_dirs else "none"
    drift = Group(
        Text(f"Orphan run dirs: {len(summary.orphan_run_dirs)}"),
        Text(orphan_runs),
        Text(""),
        Text(f"Registry entries without run dirs: {len(summary.missing_run_dirs)}"),
        Text(missing_runs),
    )
    return Panel(drift, border_style="yellow", title="Registry / Disk Drift")


def _operator_hints_panel(summary: DashboardSummary, registry_path: Path, run_root: Path) -> Panel:
    if not summary.runs:
        hints = Group(
            Text("Next move:"),
            Text("- Train baseline: bc-mlops train --config configs/train.yaml"),
            Text("- Compare runs: bc-mlops compare --registry artifacts/registry.json"),
            Text("- Open interactive deck: bc-mlops dashboard --interactive"),
        )
        return Panel(hints, border_style="cyan", title="Operator Hints")

    missing_cards = sum("missing" in status.card_status for status in summary.artifact_statuses)
    unhealthy_runs = sum(status_has_issues(status) for status in summary.artifact_statuses)
    best_run_name = str(summary.best_run["run_name"]) if summary.best_run is not None else None
    operator_actions = _operator_action_lines(best_run_name)
    hints = Group(
        Text(f"Registry: {registry_path}"),
        Text(f"Run root: {run_root}"),
        Text(f"Runs missing model cards: {missing_cards}"),
        Text(f"Runs with any artifact issue: {unhealthy_runs}"),
        Text(f"Orphan run dirs: {len(summary.orphan_run_dirs)}"),
        Text(f"Registry entries without run dirs: {len(summary.missing_run_dirs)}"),
        Text(f"Validate champion: {operator_actions[0]}"),
        Text(f"Report champion: {operator_actions[1]}"),
        Text("Interactive mode: bc-mlops dashboard --interactive"),
    )
    return Panel(hints, border_style="cyan", title="Operator Hints")


def _operator_action_lines(run_name: str | None) -> tuple[str, str]:
    if not run_name:
        return (
            "bc-mlops validate --metrics <run_dir>/metrics.json --gates configs/quality_gates.yaml",
            "bc-mlops report --run-dir <run_dir> --output <run_dir>/MODEL_CARD.md",
        )

    run_dir = Path("artifacts/runs") / run_name
    return (
        "bc-mlops validate --metrics "
        f"{run_dir / 'metrics.json'} --gates configs/quality_gates.yaml",
        f"bc-mlops report --run-dir {run_dir} --output {run_dir / 'MODEL_CARD.md'}",
    )
