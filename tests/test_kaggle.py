import json
from pathlib import Path

import pytest

from bc_mlops_showcase.cli import main
from bc_mlops_showcase.kaggle import pull_kaggle_resource


class FakeKaggleApi:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def authenticate(self) -> None:
        self.calls.append(("authenticate",))

    def dataset_download_files(
        self,
        dataset: str,
        *,
        path: str,
        unzip: bool,
        force: bool,
        quiet: bool,
    ) -> None:
        self.calls.append(
            (
                "dataset_download_files",
                dataset,
                path,
                unzip,
                force,
                quiet,
            )
        )
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "breast-cancer.csv").write_text("target,feature\n1,0.5\n")

    def competition_download_file(
        self,
        competition: str,
        file_name: str,
        *,
        path: str,
        force: bool,
        quiet: bool,
    ) -> None:
        self.calls.append(
            (
                "competition_download_file",
                competition,
                file_name,
                path,
                force,
                quiet,
            )
        )
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / file_name).write_text("PassengerId,Survived\n1,0\n")


def test_pull_kaggle_dataset_downloads_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_api = FakeKaggleApi()
    monkeypatch.setattr("bc_mlops_showcase.kaggle.build_kaggle_api", lambda: fake_api)

    result = pull_kaggle_resource(
        resource_type="dataset",
        handle="merlinswd/breast-cancer-demo",
        output_dir=tmp_path / "datasets",
    )

    assert result.resource_type == "dataset"
    assert result.handle == "merlinswd/breast-cancer-demo"
    assert result.download_dir == tmp_path / "datasets"
    assert result.downloaded_files == ["breast-cancer.csv"]
    assert fake_api.calls == [
        (
            "dataset_download_files",
            "merlinswd/breast-cancer-demo",
            str(tmp_path / "datasets"),
            True,
            False,
            False,
        ),
    ]


def test_pull_kaggle_competition_downloads_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_api = FakeKaggleApi()
    monkeypatch.setattr("bc_mlops_showcase.kaggle.build_kaggle_api", lambda: fake_api)

    result = pull_kaggle_resource(
        resource_type="competition",
        handle="titanic",
        output_dir=tmp_path / "competitions",
        file_name="train.csv",
        unzip=False,
    )

    assert result.resource_type == "competition"
    assert result.handle == "titanic"
    assert result.file_name == "train.csv"
    assert result.downloaded_files == ["train.csv"]
    assert fake_api.calls == [
        (
            "competition_download_file",
            "titanic",
            "train.csv",
            str(tmp_path / "competitions"),
            False,
            False,
        ),
    ]


def test_pull_kaggle_resource_rejects_unknown_resource_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported Kaggle resource type"):
        pull_kaggle_resource(
            resource_type="not-a-real-kind",
            handle="whatever",
            output_dir=tmp_path,
        )


def test_cli_kaggle_pull_command_prints_json_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_pull(**kwargs: object):
        assert kwargs == {
            "resource_type": "dataset",
            "handle": "merlinswd/breast-cancer-demo",
            "output_dir": tmp_path / "downloads",
            "file_name": None,
            "unzip": True,
            "force": False,
        }
        return type(
            "FakeResult",
            (),
            {
                "summary": lambda self: {
                    "resource_type": "dataset",
                    "handle": "merlinswd/breast-cancer-demo",
                    "download_dir": str(tmp_path / "downloads"),
                    "file_name": None,
                    "downloaded_files": ["breast-cancer.csv"],
                }
            },
        )()

    monkeypatch.setattr("bc_mlops_showcase.kaggle.pull_kaggle_resource", fake_pull)

    exit_code = main(
        [
            "kaggle",
            "pull",
            "--resource-type",
            "dataset",
            "--handle",
            "merlinswd/breast-cancer-demo",
            "--output-dir",
            str(tmp_path / "downloads"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resource_type"] == "dataset"
    assert payload["downloaded_files"] == ["breast-cancer.csv"]
