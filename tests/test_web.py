from http.client import HTTPConnection
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest

from PIL import Image

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
