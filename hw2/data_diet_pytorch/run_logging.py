"""Everything a run leaves behind: local files, mlflow tracking and a manifest that proves the run is complete.

Layout of a run directory:
    config.json            every parameter of the run
    environment.json       versions, GPU, git commit, command line
    epoch_metrics.csv      one row per epoch (rewritten after every epoch, so a crash loses nothing)
    run_manifest.json      list of expected files with sizes and hashes; status "complete" only if nothing is missing
plus whatever the training loop and score code register through `RunRecorder.expect_file`.
"""
import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

HW2_DIRECTORY = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_URI = f"sqlite:///{HW2_DIRECTORY / 'mlflow.db'}"
MANIFEST_FILE_NAME = "run_manifest.json"
MAX_MLFLOW_PARAMETER_LENGTH = 4000


def current_time_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(*arguments: str) -> Optional[str]:
    try:
        result = subprocess.run(["git", *arguments], cwd=HW2_DIRECTORY, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment_info() -> dict:
    import mlflow
    info = {
        "started_utc": current_time_utc(),
        "command_line": " ".join(sys.argv),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "mlflow": mlflow.__version__,
        "git_commit": _run_git("rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(_run_git("status", "--porcelain")),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def build_run_manifest(run_directory: Path, expected_files: set) -> dict:
    """Check every expected file (exists, non-empty, no NaN in arrays) and record size and hash."""
    files, problems = {}, []
    for relative_path in sorted(expected_files):
        path = run_directory / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing or empty: {relative_path}")
            continue
        entry = {"bytes": path.stat().st_size, "sha256": sha256_of_file(path)}
        if path.suffix == ".npy":
            array = np.load(path)
            entry.update(shape=list(array.shape), dtype=str(array.dtype))
            if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any():
                problems.append(f"NaN values in {relative_path}")
        files[relative_path] = entry
    manifest = {"created_utc": current_time_utc(), "status": "incomplete" if problems else "complete",
                "problems": problems, "files": files}
    _atomic_write_text(run_directory / MANIFEST_FILE_NAME, json.dumps(manifest, indent=1))
    return manifest


def verify_run_directory(run_directory: Path, check_hashes: bool = False) -> list:
    """Return a list of problems (empty = the run is complete and its files are intact)."""
    manifest_path = run_directory / MANIFEST_FILE_NAME
    if not manifest_path.is_file():
        return ["no run_manifest.json"]
    manifest = json.loads(manifest_path.read_text())
    problems = []
    if manifest["status"] != "complete":
        problems = [f"manifest status is '{manifest['status']}'"] + list(manifest["problems"])
    for relative_path, entry in manifest["files"].items():
        path = run_directory / relative_path
        if not path.is_file():
            problems.append(f"missing: {relative_path}")
        elif path.stat().st_size != entry["bytes"]:
            problems.append(f"size changed: {relative_path}")
        elif check_hashes and sha256_of_file(path) != entry["sha256"]:
            problems.append(f"hash changed: {relative_path}")
    return problems


class RunRecorder:
    """Writes config/environment/metrics to disk and, optionally, to an mlflow run."""

    def __init__(self, run_directory: Path, parameters: dict, mlflow_experiment: Optional[str] = None,
                 mlflow_run_name: Optional[str] = None, mlflow_tags: Optional[dict] = None):
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.epoch_rows: list = []
        self.expected_files = {"config.json", "environment.json", "epoch_metrics.csv"}
        self.mlflow_run_id = None
        self.environment = collect_environment_info()
        self.write_json("config.json", parameters)
        self.write_json("environment.json", self.environment)
        if mlflow_experiment:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(mlflow_experiment)
            run = mlflow.start_run(run_name=mlflow_run_name, tags=mlflow_tags)
            self.mlflow_run_id = run.info.run_id
            mlflow.log_params({key: self._as_mlflow_parameter(value) for key, value in parameters.items()})
            mlflow.log_params({f"env_{key}": self._as_mlflow_parameter(value) for key, value in self.environment.items()})
            mlflow.log_artifact(str(self.run_directory / "config.json"), "run_files")
            mlflow.log_artifact(str(self.run_directory / "environment.json"), "run_files")

    @staticmethod
    def _as_mlflow_parameter(value) -> str:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return text[:MAX_MLFLOW_PARAMETER_LENGTH]

    def write_json(self, relative_path: str, content) -> Path:
        path = self.run_directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(content, indent=1, default=str))
        return path

    def expect_file(self, relative_path: str) -> None:
        self.expected_files.add(relative_path)

    def log_epoch(self, epoch: int, row: dict) -> None:
        """Append a row to epoch_metrics.csv (rewritten atomically) and send numeric values to mlflow."""
        self.epoch_rows.append(row)
        columns = list(dict.fromkeys(key for existing in self.epoch_rows for key in existing))
        path = self.run_directory / "epoch_metrics.csv"
        temporary = path.with_suffix(".csv.tmp")
        with open(temporary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(self.epoch_rows)
        temporary.replace(path)
        if self.mlflow_run_id:
            import mlflow
            numeric = {key: float(value) for key, value in row.items()
                       if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value)}
            mlflow.log_metrics(numeric, step=epoch)

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        if self.mlflow_run_id:
            import mlflow
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)

    def log_artifact(self, relative_path: str, artifact_folder: str = "run_files") -> None:
        if self.mlflow_run_id:
            import mlflow
            mlflow.log_artifact(str(self.run_directory / relative_path), artifact_folder)

    def log_figure(self, figure, artifact_file: str) -> None:
        if self.mlflow_run_id:
            import mlflow
            mlflow.log_figure(figure, artifact_file)

    def finish(self, status: str = "complete", final_metrics: Optional[dict] = None) -> dict:
        manifest = build_run_manifest(self.run_directory, self.expected_files)
        if status != "complete":
            manifest["status"] = status
            _atomic_write_text(self.run_directory / MANIFEST_FILE_NAME, json.dumps(manifest, indent=1))
        if self.mlflow_run_id:
            import mlflow
            if final_metrics:
                mlflow.log_metrics({key: float(value) for key, value in final_metrics.items()})
            mlflow.set_tag("manifest_status", manifest["status"])
            mlflow.log_artifact(str(self.run_directory / "epoch_metrics.csv"), "run_files")
            mlflow.log_artifact(str(self.run_directory / MANIFEST_FILE_NAME), "run_files")
            mlflow.end_run("FINISHED" if status == "complete" else "FAILED")
            self.mlflow_run_id = None
        return manifest
