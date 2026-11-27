"""Shared on-disk history for the command line and local web app."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ._locking import LockBusy, process_lock


ROLES = ("front", "right", "left", "back")
SIDES = {"right": "droite", "left": "gauche", "back": "dos"}
TIERS = ("draft", "standard", "high", "max")
ACTIVE_STATUSES = ("queued", "running")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_LEGACY_MODEL = re.compile(r"model_(\d+)\.glb", re.IGNORECASE)


class DeletionConflict(RuntimeError):
    pass


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _inside(root: Path, name: str) -> Path | None:
    try:
        path = (root / name).resolve()
        return path if path.is_relative_to(root) and path.is_file() else None
    except (OSError, ValueError):
        return None


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        for attempt in range(4):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.015 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _companions(model: Path) -> dict[str, Path]:
    return {"model": model, "ai": model.with_suffix(".ai.glb"),
            "thumbnail": model.with_suffix(".png")}


def _original_name(name: str, role: str) -> str:
    return Path(name.replace("\\", "/")).name or role


class RunHistory:
    def __init__(self, runtime_root: Path, legacy_dir: Path | None = None):
        self.runtime_root = Path(runtime_root).resolve()
        self.root = self.runtime_root / "library" / "runs"
        self.legacy_dir = Path(legacy_dir).resolve() if legacy_dir is not None else None
        self._legacy_hashes: dict[Path, tuple[tuple[int, int, int, int], str]] = {}

    def _directory(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run ID.")
        directory = (self.root / run_id).resolve()
        if directory.parent != self.root.resolve() or directory.name != run_id:
            raise ValueError("Invalid run directory.")
        return directory

    def create_run(self, *, origin: str, sources: dict[str, tuple[str, bytes | Path]],
                   tier: str | None = None, seed: int | None = None,
                   projection: bool | None = None) -> dict:
        if set(sources) not in ({"front"}, set(ROLES)):
            raise ValueError("Provide one front photo or all four views.")
        run_id = uuid4().hex
        directory = self._directory(run_id)
        directory.mkdir(parents=True)
        try:
            (directory / "sources").mkdir()
            (directory / "result").mkdir()
            saved = {}
            for role in ROLES:
                if role not in sources:
                    continue
                name, data = sources[role]
                name = _original_name(name, role)
                suffix = Path(name).suffix.lower()
                if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
                    suffix = ""
                relative = f"sources/{role}{suffix}"
                if isinstance(data, Path):
                    shutil.copyfile(data, directory / relative)
                else:
                    (directory / relative).write_bytes(data)
                saved[role] = {"name": name, "path": relative}
                if isinstance(data, Path):
                    saved[role]["original_path"] = str(data.resolve())

            now = _timestamp()
            record = {
                "id": run_id, "origin": origin, "status": "queued",
                "created_at": now, "updated_at": now, "tier": tier,
                "requested_tier": tier, "seed": seed, "projection": projection,
                "progress": 0.0, "stage": "Queued", "error": None,
                "sources": saved, "assets": {}, "stats": {},
            }
            _atomic_json(directory / "run.json", record)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return record

    def _read_managed(self, run_id: str) -> dict | None:
        try:
            path = _inside(self._directory(run_id), "run.json")
            if path is not None:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict) and record.get("id") == run_id:
                    return record
        except (OSError, ValueError):
            pass
        return None

    def get(self, run_id: str) -> dict | None:
        if _RUN_ID.fullmatch(run_id):
            return self._read_managed(run_id)
        if re.fullmatch(r"legacy-[0-9a-f]{32}", run_id):
            return next((run for run in self.list_runs() if run["id"] == run_id), None)
        return None

    def _managed_runs(self) -> list[dict]:
        runs = []
        for directory in self.root.iterdir() if self.root.is_dir() else ():
            if _RUN_ID.fullmatch(directory.name):
                record = self._read_managed(directory.name)
                if record is not None:
                    runs.append(record)
        return runs

    def list_runs(self) -> list[dict]:
        runs = self._managed_runs()
        if self.legacy_dir is not None and self.legacy_dir.is_dir():
            exports: dict[str, set[str]] = {}
            for record in runs:
                export = record.get("export")
                if export:
                    exports.setdefault(export["path"], set()).add(export["sha256"])
            for model in self.legacy_dir.iterdir():
                match = _LEGACY_MODEL.fullmatch(model.name)
                if match is None or _inside(self.legacy_dir, model.name) != model:
                    continue
                try:
                    if str(model) in exports and self._legacy_hash(model) in exports[str(model)]:
                        continue
                    runs.append(self._legacy_record(model, int(match[1])))
                except OSError:
                    continue
        return sorted(runs, key=lambda run: run["created_at"], reverse=True)

    def _legacy_hash(self, model: Path) -> str:
        stat = model.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        previous = self._legacy_hashes.get(model)
        if previous is not None and previous[0] == signature:
            return previous[1]
        digest = _sha256(model)
        self._legacy_hashes[model] = (signature, digest)
        return digest

    def _legacy_record(self, model: Path, seed: int) -> dict:
        stat = model.stat()
        stamp = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="milliseconds")
        version = (str(model), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        identity = hashlib.sha256(repr(version).encode("utf-8")).hexdigest()[:32]
        return {
            "id": f"legacy-{identity}", "origin": "legacy", "status": "succeeded",
            "created_at": stamp, "updated_at": stamp, "tier": None,
            "requested_tier": None, "seed": seed, "projection": None,
            "progress": 1.0, "stage": "Done", "error": None, "sources": {},
            "assets": {key: path.name for key, path in _companions(model).items()
                       if _inside(self.legacy_dir, path.name) == path},
            "stats": {},
        }

    def update_run(self, run_id: str, **fields) -> dict:
        record = self._read_managed(run_id)
        if record is None:
            raise ValueError("Run not found.")
        record.update(fields, updated_at=_timestamp())
        _atomic_json(self._directory(run_id) / "run.json", record)
        return record

    def asset(self, run_id: str, key: str) -> Path | None:
        record = self.get(run_id)
        if record is None:
            return None
        if key in ROLES:
            relative = record["sources"].get(key, {}).get("path")
        else:
            relative = record["assets"].get(key)
        if not relative:
            return None
        directory = self.legacy_dir if record["origin"] == "legacy" else self._directory(run_id)
        return _inside(directory, relative)

    def log_path(self, run_id: str) -> Path:
        return self._directory(run_id) / "run.log"

    def cli_lock(self, run_id: str):
        return process_lock(self._directory(run_id) / "cli.lock")

    def cache_activity_lock(self, *, wait: bool = False):
        return process_lock(self.root / ".cache-activity.lock", wait=wait)

    def cli_generation_active(self) -> bool:
        for record in self._managed_runs():
            if record["origin"] != "cli":
                continue
            run_id = record["id"]
            if not (self._directory(run_id) / "cli.lock").is_file():
                continue
            try:
                with self.cli_lock(run_id):
                    pass
            except LockBusy:
                return True
        return False

    def recover_cli_runs(self) -> None:
        for record in self.list_runs():
            if record["origin"] != "cli" or record["status"] != "running":
                continue
            run_id = record["id"]
            try:
                with self.cli_lock(run_id):
                    current = self._read_managed(run_id)
                    if current is not None and current["status"] == "running":
                        self.update_run(run_id, status="interrupted", stage="Interrupted",
                                        error="The CLI process stopped during generation.")
            except LockBusy:
                continue

    def result_dir(self, run_id: str) -> Path:
        return self._directory(run_id) / "result"

    def source_paths(self, run_id: str) -> dict[str, Path]:
        return {role: path for role in ROLES if (path := self.asset(run_id, role)) is not None}

    def record_result(self, run_id: str, result) -> dict:
        directory = self._directory(run_id)
        assets = {}
        for key, path in (("model", result.glb_path), ("ai", result.glb_ai),
                          ("thumbnail", result.thumbnail)):
            if path is not None:
                relative = Path(path).resolve().relative_to(directory).as_posix()
                if _inside(directory, relative) is not None:
                    assets[key] = relative
        if "model" not in assets:
            raise FileNotFoundError("The generated GLB is missing.")
        return self.update_run(
            run_id, status="succeeded", stage="Done", progress=1.0,
            tier=result.detail, seed=result.seed, assets=assets,
            stats={"faces": result.faces, "vertices": result.vertices,
                   "duration_s": round(result.duration_s, 1),
                   "peak_vram_gb": round(result.peak_vram_gb, 2)},
        )

    def remember_export(self, run_id: str, model_path: Path) -> None:
        archived = self.asset(run_id, "model")
        if archived is None:
            raise FileNotFoundError("The archived GLB is missing.")
        self.update_run(run_id, export={"path": str(model_path.resolve()),
                                        "sha256": _sha256(archived)})

    def _export_lock(self, output: Path):
        path = os.path.normcase(str(output.resolve()))
        key = hashlib.sha256(path.encode("utf-8")).hexdigest()
        return process_lock(self.root / "export-locks" / key, wait=True)

    def export_run(self, run_id: str, destination: Path) -> Path:
        run = self.get(run_id)
        model = self.asset(run_id, "model")
        if run is None or model is None:
            raise FileNotFoundError("The archived GLB is missing.")
        output = destination / model.name
        try:
            with self._export_lock(output):
                retired = self._retired_sidecars(output, run["assets"])
                staged = []
                try:
                    files = sorted(self.result_dir(run_id).iterdir(), key=lambda path: path == model)
                    for source in files:
                        if source.is_file():
                            temporary = destination / f".{uuid4().hex}.tmp"
                            staged.append((temporary, destination / source.name))
                            shutil.copyfile(source, temporary)
                    for temporary, target in staged:
                        temporary.replace(target)
                    for path in retired:
                        path.unlink(missing_ok=True)
                finally:
                    for temporary, _ in staged:
                        temporary.unlink(missing_ok=True)
                self.remember_export(run_id, output)
        except LockBusy as exc:
            raise OSError(f"Another process is exporting {output.name}.") from exc
        return output

    def delete_run(self, run_id: str) -> bool:
        """Delete an inactive run and its still-owned result files."""
        archive_root = self.root.resolve()
        if not archive_root.is_relative_to(self.runtime_root):
            raise DeletionConflict("The history directory is outside the configured runtime.")
        try:
            with process_lock(archive_root / ".delete.lock"):
                record = self.get(run_id)
                if record is None:
                    if _RUN_ID.fullmatch(run_id):
                        staged = self._staged_deletion(run_id)
                        if staged.is_dir():
                            shutil.rmtree(staged)
                            return True
                    return False
                if record["status"] in ACTIVE_STATUSES:
                    raise DeletionConflict("Wait for this generation to finish before deleting it.")
                if record["origin"] == "legacy":
                    return self._delete_legacy(record)
                return self._delete_managed(record)
        except LockBusy as exc:
            raise DeletionConflict("This generation is still in use. Try again in a moment.") from exc

    def _staged_deletion(self, run_id: str) -> Path:
        staged = self.root / f".deleted-{run_id}"
        if staged.resolve() != staged or not staged.is_relative_to(self.root.resolve()):
            raise DeletionConflict("The staged history folder is outside the run library.")
        return staged

    def _is_source_file(self, path: Path, digest: str, records: list[dict]) -> bool:
        for record in records:
            for role, source in record["sources"].items():
                original_path = source.get("original_path")
                if original_path:
                    if Path(original_path) == path:
                        return True
                elif source["name"].casefold() == path.name.casefold():
                    # Older manifests kept only the source name and a copy.
                    archived = self.asset(record["id"], role)
                    if archived is not None and _sha256(archived) == digest:
                        return True
        return False

    def _delete_legacy(self, record: dict) -> bool:
        model = self.asset(record["id"], "model")
        if model is None:
            return False
        if model.is_relative_to(self.root.resolve()):
            raise DeletionConflict("Delete this model through its archived generation.")
        with self._export_lock(model):
            current = self.get(record["id"])
            if current is None:
                return False
            known_runs = self._managed_runs()
            owned = []
            for key in ("ai", "thumbnail", "model"):
                name = current["assets"].get(key)
                if not name or (path := _inside(self.legacy_dir, name)) != self.legacy_dir / name:
                    continue
                digest = _sha256(path)
                shared = any(run.get("export") == {"path": str(path), "sha256": digest}
                             for run in known_runs)
                if not shared and not self._is_source_file(path, digest, known_runs):
                    owned.append(path)
                elif key == "model":
                    raise DeletionConflict("This model is used by another history entry.")
            for path in owned:
                path.unlink(missing_ok=True)
        return True

    def _deletable_exports(self, record: dict, output: Path) -> list[Path]:
        if output.is_relative_to(self.root.resolve()) or not output.is_file() or output.resolve() != output:
            return []
        archived_model = self.asset(record["id"], "model")
        export = record["export"]
        if (archived_model is None or _sha256(output) != export["sha256"]
                or _sha256(archived_model) != export["sha256"]):
            return []
        records = self._managed_runs()
        others = [item.get("export") for item in records if item["id"] != record["id"]]
        if export in others:
            return []

        paths = _companions(output)
        owned = []
        for key in ("ai", "thumbnail", "model"):
            path = paths[key]
            archived = self.asset(record["id"], key)
            if archived is None or not path.is_file() or path.resolve() != path:
                continue
            digest = export["sha256"] if key == "model" else _sha256(path)
            if (key != "model" and digest != _sha256(archived)) or self._is_source_file(path, digest, records):
                continue
            if {"path": str(path), "sha256": digest} not in others:
                owned.append(path)
        return owned

    def _delete_managed(self, record: dict) -> bool:
        run_id = record["id"]
        directory = self._directory(run_id)
        staged = self._staged_deletion(run_id)
        if record["origin"] == "cli":
            # The CLI publishes its export after writing the terminal status.
            # Release the lock handle before renaming the directory on Windows.
            with self.cli_lock(run_id):
                record = self._read_managed(run_id)
                if record is None:
                    return False
                if record["status"] in ACTIVE_STATUSES:
                    raise DeletionConflict("Wait for this generation to finish before deleting it.")

        export = record.get("export")
        output = Path(export["path"]) if export else None
        lock = self._export_lock(output) if output is not None else nullcontext()
        with lock:
            files = self._deletable_exports(record, output) if output is not None else []
            directory.rename(staged)
            try:
                for path in files:
                    path.unlink(missing_ok=True)
            except OSError:
                staged.rename(directory)
                raise
        shutil.rmtree(staged)
        return True

    def _retired_sidecars(self, output: Path, assets: dict) -> list[Path]:
        # Remove only old companion files whose exact bytes are already in history.
        candidates = {key: path for key, path in _companions(output).items()
                      if key not in assets and path.is_file()}
        if not candidates or not output.is_file():
            return []
        previous_export = {"path": str(output.resolve()), "sha256": _sha256(output)}
        for row in self.list_runs():
            if row.get("export") == previous_export:
                return [path for key, path in candidates.items()
                        if (archived := self.asset(row["id"], key)) is not None
                        and _sha256(archived) == _sha256(path)]
        return []

    def import_run(self, model: Path, sources: dict[str, tuple[str, bytes | Path]], *,
                   tier: str | None = None) -> dict:
        model = model.resolve(strict=True)
        if model.suffix.lower() != ".glb" or not model.is_file():
            raise ValueError("Select a GLB model to import.")
        try:
            with self._export_lock(model):
                return self._import_run(model, sources, tier)
        except LockBusy as exc:
            raise OSError(f"Another operation is using {model.name}.") from exc

    def _import_run(self, model: Path, sources: dict[str, tuple[str, bytes | Path]], tier: str | None) -> dict:
        with model.open("rb") as stream:
            if stream.read(8) != b"glTF\x02\x00\x00\x00":
                raise ValueError("The selected file is not a glTF 2.0 GLB model.")
        stamp = datetime.fromtimestamp(model.stat().st_mtime, UTC).isoformat(timespec="milliseconds")
        export = {"path": str(model), "sha256": _sha256(model)}
        for run in self.list_runs():
            if (run["origin"] == "import" and run.get("export") == export
                    and run["tier"] == tier and self._same_sources(run, sources)):
                return run
        match = _LEGACY_MODEL.fullmatch(model.name)
        record = self.create_run(origin="import", sources=sources, tier=tier,
                                 seed=int(match[1]) if match else None)
        run_id = record["id"]
        assets = {}
        try:
            for key, path in _companions(model).items():
                if path.is_file():
                    destination = self.result_dir(run_id) / path.name
                    shutil.copyfile(path, destination)
                    assets[key] = f"result/{path.name}"
            return self.update_run(run_id, status="succeeded", stage="Done", progress=1.0,
                                   created_at=stamp, assets=assets, export=export)
        except BaseException:
            shutil.rmtree(self._directory(run_id), ignore_errors=True)
            raise

    def _same_sources(self, run: dict, sources: dict[str, tuple[str, bytes | Path]]) -> bool:
        if sources.keys() != run["sources"].keys():
            return False
        for role, (name, data) in sources.items():
            archived = self.asset(run["id"], role)
            if archived is None or _original_name(name, role) != run["sources"][role]["name"]:
                return False
            digest = _sha256(data) if isinstance(data, Path) else hashlib.sha256(data).hexdigest()
            if _sha256(archived) != digest:
                return False
        return True
