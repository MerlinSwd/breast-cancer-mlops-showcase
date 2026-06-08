"""Helpers for downloading Kaggle datasets and competition files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class KaggleApiProtocol(Protocol):
    """Subset of the Kaggle API used by this project."""

    def authenticate(self) -> None: ...

    def dataset_download_files(
        self,
        dataset: str,
        *,
        path: str,
        unzip: bool,
        force: bool,
        quiet: bool,
    ) -> None: ...

    def dataset_download_file(
        self,
        dataset: str,
        file_name: str,
        *,
        path: str,
        force: bool,
        quiet: bool,
    ) -> None: ...

    def competition_download_files(
        self,
        competition: str,
        *,
        path: str,
        force: bool,
        quiet: bool,
    ) -> None: ...

    def competition_download_file(
        self,
        competition: str,
        file_name: str,
        *,
        path: str,
        force: bool,
        quiet: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class KaggleResourceSpec:
    """Dispatch metadata for one supported Kaggle resource type."""

    resource_type: str
    label: str
    collection_method: str
    single_file_method: str
    supports_unzip: bool


@dataclass(frozen=True, slots=True)
class KagglePullResult:
    """Structured summary of a Kaggle download."""

    resource_type: str
    handle: str
    download_dir: Path
    file_name: str | None
    downloaded_files: list[str]
    unzip: bool
    force: bool

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""

        return {
            "resource_type": self.resource_type,
            "handle": self.handle,
            "download_dir": str(self.download_dir),
            "file_name": self.file_name,
            "downloaded_files": self.downloaded_files,
            "unzip": self.unzip,
            "force": self.force,
        }


KAGGLE_RESOURCE_SPECS: dict[str, KaggleResourceSpec] = {
    "dataset": KaggleResourceSpec(
        resource_type="dataset",
        label="Kaggle dataset",
        collection_method="dataset_download_files",
        single_file_method="dataset_download_file",
        supports_unzip=True,
    ),
    "competition": KaggleResourceSpec(
        resource_type="competition",
        label="Kaggle competition",
        collection_method="competition_download_files",
        single_file_method="competition_download_file",
        supports_unzip=False,
    ),
}


def get_kaggle_resource_spec(resource_type: str) -> KaggleResourceSpec:
    """Return the registered dispatch metadata for a Kaggle resource type."""

    try:
        return KAGGLE_RESOURCE_SPECS[resource_type]
    except KeyError as exc:
        raise ValueError(f"unsupported Kaggle resource type: {resource_type}") from exc


def build_kaggle_api() -> KaggleApiProtocol:
    """Construct an authenticated Kaggle API client."""

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional env state
        raise RuntimeError(
            "Kaggle support requires the 'kaggle' package. Install it with `uv add kaggle` "
            "or `uv sync` after pulling the latest dependencies."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def _list_downloaded_files(download_dir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(download_dir)) for path in download_dir.rglob("*") if path.is_file()
    )


def pull_kaggle_resource(
    *,
    resource_type: str,
    handle: str,
    output_dir: str | Path,
    file_name: str | None = None,
    unzip: bool = True,
    force: bool = False,
) -> KagglePullResult:
    """Download a Kaggle dataset or competition bundle into a local directory."""

    spec = get_kaggle_resource_spec(resource_type)
    download_dir = Path(output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    api = build_kaggle_api()

    if file_name:
        download_method = getattr(api, spec.single_file_method)
        download_method(
            handle,
            file_name,
            path=str(download_dir),
            force=force,
            quiet=False,
        )
    else:
        download_method = getattr(api, spec.collection_method)
        kwargs = {
            "path": str(download_dir),
            "force": force,
            "quiet": False,
        }
        if spec.supports_unzip:
            kwargs["unzip"] = unzip
        download_method(handle, **kwargs)

    return KagglePullResult(
        resource_type=resource_type,
        handle=handle,
        download_dir=download_dir,
        file_name=file_name,
        downloaded_files=_list_downloaded_files(download_dir),
        unzip=unzip,
        force=force,
    )
