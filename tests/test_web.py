from http.client import HTTPConnection
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from PIL import Image

from localmesh_engine._locking import process_lock
from localmesh_engine.run_history import RunHistory
from localmesh_engine.web.server import WebRuns, make_server


FAKE_WORKER = """
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from localmesh_engine.run_history import RunHistory

history = RunHistory(Path(sys.argv[1]))
run_id = sys.argv[2]
history.update_run(run_id, status='running', stage='Generating shape', progress=0.42)
print('Writing a test GLB', flush=True)
if history.get(run_id)['seed'] == 10:
    raise SystemExit(10)
while not (history.runtime_root / 'release').exists():
    time.sleep(0.025)
result_dir = history.result_dir(run_id)
model = result_dir / 'model_42.glb'
model.write_bytes(b'glTF fake model')
ai = result_dir / 'model_42.ai.glb'
ai.write_bytes(b'glTF fake AI model')
history.record_result(run_id, SimpleNamespace(
    glb_path=model, glb_ai=ai, thumbnail=None, detail='draft', seed=42,
    faces=234, vertices=123, duration_s=2.25, peak_vram_gb=1.23,
))
"""


def multipart(fields: dict[str, str | tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = "localmesh-test-boundary"
    output = BytesIO()
    for key, value in fields.items():
        output.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'.encode())
        if isinstance(value, tuple):
            name, content = value
            output.write(f'; filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode())
        else:
            content = value.encode()
            output.write(b"\r\n\r\n")
        output.write(content + b"\r\n")
    output.write(f"--{boundary}--\r\n".encode())
    return output.getvalue(), f"multipart/form-data; boundary={boundary}"


class WebIntegrationTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.runtime = Path(tmp.name)
        self.legacy = self.runtime / "old"
        self.legacy.mkdir()
        self.history = RunHistory(self.runtime, self.legacy)
        self.runs = WebRuns(self.history, (sys.executable, "-B", "-u", "-c", FAKE_WORKER))
        self.server = make_server(self.runs, 0)
        thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.025})
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.runs.close)
        self.addCleanup(self.server.shutdown)
        output = BytesIO()
        Image.new("RGB", (3, 2), (25, 75, 125)).save(output, format="PNG")
        self.picture = output.getvalue()

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def post(self, fields, **headers):
        form, mime = multipart(fields)
        status, _, data = self.request("POST", "/api/runs", form, {"Content-Type": mime, **headers})
        return status, json.loads(data)

    def wait_for(self, run_id, status):
        deadline = monotonic() + 6
        while monotonic() < deadline:
            code, _, data = self.request("GET", f"/api/runs/{run_id}")
            self.assertEqual(code, 200)
            row = json.loads(data)
            if row["status"] == status:
                return row
            sleep(0.025)
        self.fail(f"Expected status {status!r}, got {row!r}")

    def test_generation_and_original_survive_restart(self):
        fields = {"front": ("образец.png", self.picture), "tier": "draft", "seed": "-1", "projection": "false"}
        status, submitted = self.post(fields)
        self.assertEqual(status, 202)
        self.assertEqual(submitted["images"][0]["name"], "образец.png")
        self.assertIs(submitted["projection"], False)
        running = self.wait_for(submitted["id"], "running")
        self.assertEqual(running["progress"], 0.42)
        self.assertEqual(self.post(fields)[0], 409)

        (self.runtime / "release").touch()
        done = self.wait_for(submitted["id"], "succeeded")
        self.assertIn("Writing a test GLB", done["log"])
        self.assertEqual(done["faces"], 234)
        self.assertEqual(done["seed"], 42)
        self.assertEqual(self.request("GET", done["model_url"]), (200, "model/gltf-binary", b"glTF fake model"))
        self.assertEqual(self.request("GET", done["ai_model_url"])[2], b"glTF fake AI model")
        self.assertEqual(self.request("GET", done["images"][0]["url"]), (200, "image/png", self.picture))
        new_history = RunHistory(self.runtime, self.legacy)
        self.assertEqual(new_history.get(done["id"])["status"], "succeeded")
        self.assertEqual(new_history.asset(done["id"], "front").read_bytes(), self.picture)

    def test_worker_failure_is_recorded(self):
        status, row = self.post({"front": ("photo.png", self.picture), "seed": "10"})
        self.assertEqual(status, 202)
        failure = self.wait_for(row["id"], "failed")
        self.assertIn("code 10", failure["error"])
        self.assertIn("Writing a test GLB", failure["log"])
        self.assertIsNone(failure["model_url"])

    def test_delete_requires_terminal_run_and_local_origin(self):
        status, row = self.post({"front": ("photo.png", self.picture)})
        self.assertEqual(status, 202)
        run_id = row["id"]
        path = f"/api/runs/{run_id}"
        self.wait_for(run_id, "running")
        self.assertEqual(self.request("DELETE", path)[0], 409)
        self.runs.close()
        self.wait_for(run_id, "interrupted")
        for headers in ({"Origin": "https://other.example"}, {"Host": "other.example"},
                        {"Sec-Fetch-Site": "cross-site"}):
            self.assertEqual(self.request("DELETE", path, headers=headers)[0], 403)
        origin = f"http://127.0.0.1:{self.server.server_port}"
        self.assertEqual(self.request("DELETE", path, headers={"Origin": origin}), (204, None, b""))
        self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("GET", row["images"][0]["url"])[0], 404)
        self.assertEqual(self.request("DELETE", path)[0], 404)
        self.assertFalse((self.history.root / run_id).exists())

    def test_clear_cache_preserves_runtime_history_and_external_overrides(self):
        cache = self.runtime / "cache"
        for folder in ("kernels", "flex_gemm", "triton", "cuda", "verification"):
            artifact = cache / folder / "nested" / "temporary.bin"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"cache")
        (cache / "temporary.tmp").write_bytes(b"cache")
        weights = self.runtime / "models" / "hf" / "weights.bin"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"models")
        other = self.runtime / "override" / "temporary.bin"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"override")
        run = self.history.create_run(origin="web", sources={"front": ("photo.png", self.picture)})

        self.assertEqual(self.request("DELETE", "/api/cache", headers={"Origin": "https://other.example"})[0], 403)
        self.assertTrue(list(cache.iterdir()))
        with patch.dict(os.environ, {"TRITON_CACHE_DIR": str(other.parent)}):
            status, _, body = self.request("DELETE", "/api/cache")
        self.assertEqual((status, json.loads(body)), (200, {"removed": 6, "failed": 0}))
        self.assertTrue(cache.is_dir())
        self.assertEqual(list(cache.iterdir()), [])
        self.assertEqual(weights.read_bytes(), b"models")
        self.assertEqual(other.read_bytes(), b"override")
        self.assertEqual(self.history.asset(run["id"], "front").read_bytes(), self.picture)
        status, _, body = self.request("DELETE", "/api/cache")
        self.assertEqual((status, json.loads(body)), (200, {"removed": 0, "failed": 0}))

    def test_clear_cache_checks_cli_registration_and_live_workers(self):
        cache = self.runtime / "cache"
        cache.mkdir()
        artifact = cache / "temporary.bin"
        artifact.write_bytes(b"cache")
        cli = self.history.create_run(origin="cli", sources={"front": ("photo.png", self.picture)})
        run_id = cli["id"]
        for lock in (self.history.cache_activity_lock(), self.history.cli_lock(run_id),
                     process_lock(self.history.root / ".web-worker.lock")):
            with lock:
                self.assertEqual(self.request("DELETE", "/api/cache")[0], 409)
                self.assertTrue(artifact.is_file())
        self.history.update_run(run_id, status="running")
        self.assertEqual(self.request("DELETE", "/api/cache")[0], 200)
        self.assertFalse(artifact.exists())

        artifact.write_bytes(b"cache")
        status, row = self.post({"front": ("photo.png", self.picture)})
        self.assertEqual(status, 202)
        self.wait_for(row["id"], "running")
        self.assertEqual(self.request("DELETE", "/api/cache")[0], 409)
        self.assertTrue(artifact.is_file())

    def test_clear_cache_reports_partial_failure_and_protects_model_overrides(self):
        cache = self.runtime / "cache"
        cache.mkdir()
        blocked = cache / "locked.bin"
        blocked.write_bytes(b"locked")
        available = cache / "available.bin"
        available.write_bytes(b"available")
        custom_weights = cache / "hf" / "models.bin"
        custom_weights.parent.mkdir()
        custom_weights.write_bytes(b"weights")
        unlink = Path.unlink

        def locked_unlink(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("File is open")
            return unlink(path, *args, **kwargs)

        with (self.assertLogs(level="WARNING"), patch.object(Path, "unlink", locked_unlink),
              patch.dict(os.environ, {"HF_HUB_CACHE": str(custom_weights.parent)})):
            status, _, body = self.request("DELETE", "/api/cache")
        result = json.loads(body)
        self.assertEqual((status, result["removed"], result["failed"]), (500, 1, 2))
        self.assertIn("Removed 1 item", result["error"])
        self.assertIn("Could not remove 2", result["error"])
        self.assertFalse(available.exists())
        self.assertEqual(custom_weights.read_bytes(), b"weights")
        self.assertEqual(blocked.read_bytes(), b"locked")

    def make_directory_link(self, link, target):
        if os.name != "nt":
            link.symlink_to(target, target_is_directory=True)
            return
        # Junctions can be created on Windows without symlink privileges.
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "$null = New-Item -ItemType Junction -Path $env:TEST_CACHE_LINK -Target $env:TEST_CACHE_TARGET"],
            env={**os.environ, "TEST_CACHE_LINK": str(link), "TEST_CACHE_TARGET": str(target)},
            capture_output=True, check=False,
        )
        if result.returncode:
            self.skipTest("Could not create a Windows junction in the temporary directory.")

    def test_clear_cache_rejects_link_at_root_and_removes_nested_link_only(self):
        cache = self.runtime / "cache"
        outside = self.runtime / "external"
        outside.mkdir()
        artifact = outside / "data.bin"
        artifact.write_bytes(b"external")
        self.make_directory_link(cache, outside)
        self.assertEqual(self.request("DELETE", "/api/cache")[0], 409)
        self.assertEqual(artifact.read_bytes(), b"external")
        if cache.is_junction():
            cache.rmdir()
        else:
            cache.unlink()

        nested = cache / "triton"
        nested.mkdir(parents=True)
        self.make_directory_link(nested / "linked", outside)
        self.assertEqual(self.request("DELETE", "/api/cache")[0], 200)
        self.assertEqual(list(cache.iterdir()), [])
        self.assertEqual(artifact.read_bytes(), b"external")

    def test_clear_cache_preserves_model_storage_redirected_inside_cache(self):
        cache = self.runtime / "cache"
        storage = cache / "redirected-models"
        storage.mkdir(parents=True)
        artifact = storage / "weights.bin"
        artifact.write_bytes(b"weights")
        self.make_directory_link(self.runtime / "models", storage)

        with self.assertLogs(level="WARNING"):
            status, _, body = self.request("DELETE", "/api/cache")
        self.assertEqual(status, 500)
        result = json.loads(body)
        self.assertEqual((result["removed"], result["failed"]), (0, 1))
        self.assertEqual(artifact.read_bytes(), b"weights")

    def test_four_views_and_shutdown(self):
        roles = ["front", "right", "left", "back"]
        status, row = self.post({role: (f"{role}.png", self.picture) for role in roles})
        self.assertEqual(status, 202)
        self.assertEqual([image["role"] for image in row["images"]], roles)
        self.assertIs(row["projection"], True)
        self.assertEqual(row["tier"], "standard")
        self.wait_for(row["id"], "running")
        self.runs.close()
        interrupted = self.wait_for(row["id"], "interrupted")
        self.assertIn("web server was stopped", interrupted["error"])
        self.assertEqual(self.request("GET", interrupted["images"][-1]["url"])[2], self.picture)

    def test_upload_and_local_origin_checks(self):
        front = {"front": ("front.png", self.picture)}
        self.assertEqual(self.post({})[0], 400)
        self.assertEqual(self.post({**front, "left": ("left.png", self.picture)})[0], 400)
        self.assertEqual(self.post({"front": ("misnamed.jpg", self.picture)})[0], 400)
        self.assertEqual(self.post({"front": ("corrupt.png", b"data")})[0], 400)
        self.assertEqual(self.post({**front, "seed": "2147483648"})[0], 400)
        self.assertEqual(self.post(front, Origin="https://other.example")[0], 403)
        self.assertEqual(self.post(front, **{"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.request("GET", "/api/runs", headers={"Host": "other.example"})[0], 403)
        self.assertEqual(self.request("GET", "/static/vendor/three/%2e%2e/%2e%2e/app.js")[0], 404)
        self.assertEqual(self.history.list_runs(), [])

    def test_legacy_model_and_recover_stopped_runs(self):
        model = self.legacy / "model_6.glb"
        model.write_bytes(b"old model")
        model.with_suffix(".ai.glb").write_bytes(b"old AI model")
        model.with_suffix(".png").write_bytes(self.picture)
        for origin in ("web", "cli"):
            row = self.history.create_run(origin=origin, sources={"front": ("original.png", self.picture)})
            self.history.update_run(row["id"], status="running")
        with self.history.cli_lock(row["id"]):
            self.runs.recover()
            self.assertEqual(self.history.get(row["id"])["status"], "running")
        self.runs.recover()
        status, _, data = self.request("GET", "/api/runs")
        self.assertEqual(status, 200)
        by_origin = {row["origin"]: row for row in json.loads(data)["runs"]}
        self.assertEqual(by_origin["web"]["status"], "interrupted")
        self.assertEqual(by_origin["cli"]["status"], "interrupted")
        legacy = by_origin["legacy"]
        self.assertEqual(legacy["images"], [])
        self.assertEqual(legacy["seed"], 6)
        self.assertEqual(self.request("GET", legacy["model_url"])[2], b"old model")
        self.assertEqual(self.request("GET", legacy["thumbnail_url"])[2], self.picture)


if __name__ == "__main__":
    unittest.main()
