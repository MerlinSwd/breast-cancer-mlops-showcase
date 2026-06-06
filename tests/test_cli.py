import json
from pathlib import Path

from bc_mlops_showcase.cli import main


def test_cli_train_command_generates_run_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment_name: cli-test
""".strip()
    )
    output_dir = tmp_path / "artifacts"

    exit_code = main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    run_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    assert run_dirs, "expected at least one run directory"
    metrics = json.loads((run_dirs[-1] / "metrics.json").read_text())
    assert metrics["f1"] >= 0.85
