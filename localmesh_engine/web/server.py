"""HTTP and background-process boundary for the local browser interface."""
from __future__ import annotations

from email import policy
from email.message import Message
from email.parser import BytesParser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import logging
import mimetypes
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
from threading import Lock, Thread
from urllib.parse import unquote, urlsplit

from ..run_history import ACTIVE_STATUSES, ROLES, TIERS, DeletionConflict, RunHistory
from .._locking import LockBusy, process_lock


STATIC_DIR = Path(__file__).with_name("static").resolve()
STATIC_FILES = {"/": "index.html", "/static/styles.css": "styles.css",
                "/static/app.js": "app.js", "/static/viewer.js": "viewer.js"}
VENDOR_PATH = "/static/vendor/three/"
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'; "
    "object-src 'none'; img-src 'self' blob: data:; connect-src 'self' blob:; "
    "script-src 'self' 'sha256-JGmfLJikGadX+CinRFq2zpEompv1H9ab08IfiDMIEQ4='"
)
MAX_REQUEST_BYTES = 64 * 1024 ** 2
MAX_IMAGE_BYTES = 24 * 1024 ** 2
MAX_LOG_BYTES = 32 * 1024
IMAGE_SUFFIXES = {
    "PNG": (".png",), "JPEG": (".jpg", ".jpeg", ".jpe"), "WEBP": (".webp",),
    "GIF": (".gif",), "BMP": (".bmp",),
}


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status


def _upload_name(filename: str, picture: bytes, role: str) -> str:
    from PIL import Image

    if not picture or len(picture) > MAX_IMAGE_BYTES:
        raise ValueError(f"The {role} image must be between 1 byte and 24 MiB.")
    try:
        with Image.open(BytesIO(picture)) as image:
            image_format = image.format
            extensions = IMAGE_SUFFIXES.get(image_format)
            if extensions is None:
                raise ValueError(f"Choose a PNG, JPEG, WebP, GIF, or BMP {role} image.")
            if image.width * image.height > 50_000_000:
                raise ValueError(f"The {role} image must be at most 50 megapixels.")
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"The {role} file could not be read as an image.") from exc

    name = PureWindowsPath(filename).name.strip()
    if not name or len(name) > 255 or any(ord(character) < 32 for character in name):
        raise ValueError(f"The {role} image has an invalid file name.")
    if Path(name).suffix.lower() not in extensions:
        raise ValueError(f"The {role} image extension must match its {image_format} format.")
    return name


def _read_form(content_type: str, body: bytes) -> dict:
    prefix = f"Content-Type: {content_type}\r\n\r\n".encode("ascii")
    message = BytesParser(policy=policy.compat32).parsebytes(prefix + body)
    if (message.get_content_type() != "multipart/form-data" or not message.is_multipart()
            or message.defects):
        raise ValueError("Send a multipart form containing your source images.")

    sources = {}
    settings = {}
    seen = set()
    for field in message.get_payload():
        if (field.is_multipart() or field.get_content_disposition() != "form-data"
                or field.get("Content-Transfer-Encoding") is not None):
            raise ValueError("The form contains an invalid field.")
        # HTML forms send UTF-8 filenames in raw header bytes. Decode before
        # asking the email parser for the parameters so it preserves the name.
        raw = next(value for name, value in field.raw_items() if name.lower() == "content-disposition")
        disposition = Message()
        disposition["Content-Disposition"] = raw.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        key = disposition.get_param("name", header="content-disposition")
        if not isinstance(key, str) or key in seen:
            raise ValueError("The form contains a missing or duplicate field name.")
        seen.add(key)
        content = field.get_payload(decode=True)
        filename = disposition.get_filename()
        if key in ROLES:
            if not filename and not content:
                continue
            if filename is None:
                raise ValueError(f"Choose a file for the {key} image.")
            sources[key] = (_upload_name(filename, content, key), content)
        elif key in ("tier", "seed", "projection") and filename is None:
            if len(content) > 32:
                raise ValueError(f"The {key} value is too long.")
            settings[key] = content.decode("ascii").strip().lower()
        else:
            raise ValueError("The form contains an unexpected field.")

    if "front" not in sources:
        raise ValueError("Choose a front image.")
    if len(sources) not in (1, 4):
        raise ValueError("Four-view generation requires the right, left, and back images.")

    tier = settings.get("tier", "standard")
    if tier not in TIERS:
        raise ValueError("Choose an available quality tier.")
    seed_text = settings.get("seed") or "-1"
    if not re.fullmatch(r"-1|[0-9]+", seed_text) or int(seed_text) > 2 ** 31 - 1:
        raise ValueError("The seed must be -1 or an integer from 0 to 2147483647.")
    projection = settings.get("projection", "true")
    if projection not in ("true", "false"):
        raise ValueError("Choose whether to project the source images.")
    return {"sources": sources, "tier": tier, "seed": int(seed_text),
            "projection": projection == "true"}


class WebRuns:
    def __init__(self, history: RunHistory, worker_command: tuple[str, ...] | None = None):
        self.history = history
        self._worker_command = worker_command or (sys.executable, "-u", "-m", "localmesh_engine.web._worker")
        self._lock = Lock()
        self._worker_lock = history.root / ".web-worker.lock"
        self._active: subprocess.Popen | None = None
        self._active_run_id: str | None = None
        self._monitor: Thread | None = None
        self._stopping = False

    def recover(self) -> None:
        self.history.recover_cli_runs()
        try:
            with process_lock(self._worker_lock):
                for row in self.history.list_runs():
                    if row["origin"] == "web" and row["status"] in ACTIVE_STATUSES:
                        self.history.update_run(row["id"], status="interrupted", stage="Interrupted",
                                                error="The previous web server stopped during generation.")
        except LockBusy:
            # A worker left by an unclean shutdown can finish writing its result.
            pass

    def submit(self, settings: dict) -> dict:
        with self._lock:
            if self._stopping or self._active is not None:
                raise RequestError(HTTPStatus.CONFLICT, "A generation is running. Wait for it to finish.")
            try:
                with process_lock(self._worker_lock):
                    row = self.history.create_run(origin="web", **settings)
            except LockBusy as exc:
                raise RequestError(HTTPStatus.CONFLICT,
                                   "A generation is running. Wait for it to finish.") from exc

            run_id = row["id"]
            root = str(self.history.runtime_root.resolve())
            try:
                with self.history.log_path(run_id).open("ab") as log:
                    self._active = subprocess.Popen(
                        (*self._worker_command, root, run_id),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "LOCALMESH_ROOT": root, "PYTHONIOENCODING": "utf-8"},
                    )
            except OSError as exc:
                return self.history.update_run(run_id, status="failed", stage="Failed",
                                               error=f"Could not start the generation process: {exc}")
            self._active_run_id = run_id
            self._monitor = Thread(target=self._watch, args=(run_id, self._active), daemon=True)
            self._monitor.start()
            return row

    def _watch(self, run_id: str, worker: subprocess.Popen) -> None:
        code = worker.wait()
        with self._lock:
            try:
                row = self.history.get(run_id)
                if row is not None and row["status"] in ACTIVE_STATUSES:
                    if self._stopping:
                        self.history.update_run(run_id, status="interrupted", stage="Interrupted",
                                                error="The web server was stopped during generation.")
                    else:
                        self.history.update_run(
                            run_id, status="failed", stage="Failed",
                            error=f"The generation process exited without a result (code {code}). See the log.",
                        )
            except (OSError, ValueError):
                logging.exception("Could not update generation status")
            finally:
                self._active = None
                self._active_run_id = None

    def delete(self, run_id: str) -> bool:
        with self._lock:
            if run_id == self._active_run_id:
                raise DeletionConflict("Wait for this generation to finish before deleting it.")
        return self.history.delete_run(run_id)

    def close(self) -> None:
        with self._lock:
            self._stopping = True
            worker, monitor = self._active, self._monitor
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
            except ProcessLookupError:
                pass
        if monitor is not None:
            monitor.join(timeout=10)

    def public(self, row: dict, *, detail: bool = False) -> dict:
        run_id = row["id"]
        url = f"/api/runs/{run_id}/assets/"
        sources = row.get("sources", {})
        assets = row.get("assets", {})
        stats = row.get("stats", {})
        title = sources.get("front", {}).get("name") or Path(assets.get("model") or "Generation").stem
        response = {key: row.get(key) for key in (
            "id", "created_at", "status", "origin", "tier", "seed", "projection", "progress", "stage", "error",
        )}
        response.update({
            "name": title, "legacy": row["origin"] == "legacy",
            "images": [{"role": role, "name": sources[role]["name"], "url": url + role}
                       for role in ROLES if role in sources],
            "model_url": url + "model" if assets.get("model") else None,
            "ai_model_url": url + "ai" if assets.get("ai") else None,
            "thumbnail_url": url + "thumbnail" if assets.get("thumbnail") else None,
            "faces": stats.get("faces"), "duration_s": stats.get("duration_s"),
            "peak_vram_gb": stats.get("peak_vram_gb"),
        })
        if detail:
            response["log"] = self._log_tail(run_id) if row["origin"] in ("cli", "web") else ""
        return response

    def _log_tail(self, run_id: str) -> str:
        try:
            with self.history.log_path(run_id).open("rb") as log:
                size = log.seek(0, os.SEEK_END)
                log.seek(max(0, size - MAX_LOG_BYTES))
                text = log.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
                return text.replace("\r\n", "\n").replace("\r", "\n")
        except FileNotFoundError:
            return ""


class LandingHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, runs: WebRuns, **kwargs):
        self.runs = runs
        self._body_read = False
        super().__init__(*args, **kwargs)

    def log_message(self, _format, *_args):
        # Polling and asset requests would otherwise bury generation messages.
        pass

    def setup(self) -> None:
        self.request.settimeout(30)
        super().setup()

    def _validate_host(self) -> str:
        port = self.server.server_address[1]
        host = self.headers.get("Host", "").lower()
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update(("127.0.0.1", "localhost"))
        if host not in allowed:
            raise RequestError(HTTPStatus.FORBIDDEN, "Open LocalMesh using its loopback URL.")
        return host

    def _validate_origin(self) -> None:
        host = self._validate_host()
        origin = f"http://{host}"
        if (self.headers.get("Origin", origin) != origin
                or self.headers.get("Sec-Fetch-Site") not in (None, "same-origin")):
            raise RequestError(HTTPStatus.FORBIDDEN, "Send this request from the LocalMesh page.")

    def _headers(self, status: HTTPStatus, mime: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.end_headers()

    def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(payload))
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _file(self, path: Path, *, source_image: bool = False) -> None:
        if source_image:
            from PIL import Image

            try:
                with Image.open(path) as image:
                    content_type = Image.MIME.get(image.format, "application/octet-stream")
            except (OSError, ValueError, Image.DecompressionBombError):
                content_type = "application/octet-stream"
        else:
            content_type = ({".glb": "model/gltf-binary", ".js": "text/javascript", ".css": "text/css"}
                            .get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
                            or "application/octet-stream")
        try:
            stream = path.open("rb")
        except (FileNotFoundError, IsADirectoryError):
            raise RequestError(HTTPStatus.NOT_FOUND, "File not found.") from None
        with stream:
            self._headers(HTTPStatus.OK, content_type, os.fstat(stream.fileno()).st_size)
            if self.command != "HEAD":
                shutil.copyfileobj(stream, self.wfile, length=64 * 1024)

    def _get(self) -> None:
        self._validate_host()
        path = unquote(urlsplit(self.path).path)
        if path in STATIC_FILES:
            return self._file(STATIC_DIR / STATIC_FILES[path])
        if path.startswith(VENDOR_PATH):
            vendor = STATIC_DIR / "vendor" / "three"
            asset = (vendor / path[len(VENDOR_PATH):]).resolve()
            if asset.is_relative_to(vendor) and asset.suffix == ".js" and asset.is_file():
                return self._file(asset)
        elif path == "/api/runs":
            return self._json({"runs": [self.runs.public(row) for row in self.runs.history.list_runs()]})
        else:
            match = re.fullmatch(r"/api/runs/([a-zA-Z0-9_-]+)(?:/assets/([a-z]+))?", path)
            if match:
                run_id, key = match.groups()
                if key in (*ROLES, "model", "ai", "thumbnail"):
                    asset = self.runs.history.asset(run_id, key)
                    if asset is not None:
                        return self._file(asset, source_image=key in ROLES)
                elif key is None:
                    row = self.runs.history.get(run_id)
                    if row is not None:
                        return self._json(self.runs.public(row, detail=True))
        raise RequestError(HTTPStatus.NOT_FOUND, "Page or file not found.")

    def do_GET(self) -> None:
        self._handle(self._get)

    def do_HEAD(self) -> None:
        self._handle(self._get)

    def _post(self) -> None:
        self._validate_origin()
        if self.path != "/api/runs":
            raise RequestError(HTTPStatus.NOT_FOUND, "Page not found.")
        if self.headers.get_content_type() != "multipart/form-data":
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send a multipart form.")
        length = self.headers.get("Content-Length", "")
        if not re.fullmatch(r"[0-9]{1,10}", length):
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "A valid Content-Length is required.")
        if int(length) > MAX_REQUEST_BYTES:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Keep the upload below 64 MiB.")
        body = self.rfile.read(int(length))
        self._body_read = True
        if len(body) != int(length):
            raise RequestError(HTTPStatus.BAD_REQUEST, "The upload ended before all data arrived.")
        try:
            settings = _read_form(self.headers["Content-Type"], body)
            row = self.runs.submit(settings)
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._json(self.runs.public(row, detail=True), HTTPStatus.ACCEPTED)

    def do_POST(self) -> None:
        self._handle(self._post)

    def _delete(self) -> None:
        self._validate_origin()
        path = unquote(urlsplit(self.path).path)
        match = re.fullmatch(r"/api/runs/((?:legacy-)?[0-9a-f]{32})", path)
        if not match:
            raise RequestError(HTTPStatus.NOT_FOUND, "Generation not found.")
        try:
            deleted = self.runs.delete(match[1])
        except DeletionConflict as exc:
            raise RequestError(HTTPStatus.CONFLICT, str(exc)) from exc
        except OSError as exc:
            logging.exception("Could not delete generation %s", match[1])
            raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR,
                               "Could not delete all saved files. Close any open files and try again.") from exc
        if not deleted:
            raise RequestError(HTTPStatus.NOT_FOUND, "Generation not found.")
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_DELETE(self) -> None:
        self._handle(self._delete)

    def _handle(self, handler) -> None:
        try:
            handler()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except RequestError as exc:
            try:
                self._json({"error": str(exc)}, exc.status)
                if self.command == "POST" and not self._body_read:
                    length = self.headers.get("Content-Length", "")
                    if re.fullmatch(r"[0-9]{1,5}", length):
                        # Closing with a small request body unread resets the
                        # connection on Windows and can swallow the response.
                        self.connection.settimeout(0.2)
                        self.rfile.read(int(length))
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        except (OSError, ValueError):
            logging.exception("LocalMesh request failed")
            self._json({"error": "The server could not complete this request. See the terminal."},
                       HTTPStatus.INTERNAL_SERVER_ERROR)


def make_server(runs: WebRuns, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), partial(LandingHandler, runs=runs))
