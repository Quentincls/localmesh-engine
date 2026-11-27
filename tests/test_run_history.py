import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import localmesh_engine
from localmesh_engine.__main__ import main
from localmesh_engine.progress import english_stage
from localmesh_engine.run_history import DeletionConflict, RunHistory


GLB = b"glTF\x02\x00\x00\x00"


class FakeEngine:
    def generate(self, settings, output, progress):
        progress("encodage de la forme", 0.7)
        model = output / f"model_{settings.seed}.glb"
        model.write_bytes(GLB + settings.images[0].read_bytes())
        ai = model.with_suffix(".ai.glb") if settings.projection_texture else None
        if ai:
            ai.write_bytes(GLB + b"ai")
        return SimpleNamespace(glb_path=model, glb_ai=ai, thumbnail=None,
                               detail=settings.detail, seed=settings.seed,
                               faces=12, vertices=8, duration_s=0.1, peak_vram_gb=0.1)


class RunHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base = Path(directory.name)
        self.runtime = base / "LocalMesh"
        self.out = base / "out"
        self.out.mkdir()
        self.photo = base / "photo.png"
        self.history = RunHistory(self.runtime, self.out)

    def run_cli(self, arguments):
        config = SimpleNamespace(RUNTIME_ROOT=self.runtime,
                                 gpu_report=lambda: SimpleNamespace(name="Fake GPU", total_vram_gb=8),
                                 palier_tenable=lambda _tier, _vram: True)
        pipeline = SimpleNamespace(Engine=FakeEngine, GenerateSettings=SimpleNamespace)
        stdout, stderr = io.StringIO(), io.StringIO()
        with (patch.dict("sys.modules", {"localmesh_engine.pipeline": pipeline}),
              patch.dict(localmesh_engine.__dict__, {"config": config}),
              contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
            code = main([str(self.photo), *arguments, "--to", str(self.out)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_archives_repeated_seeds_and_hides_export_alias(self):
        self.photo.write_bytes(b"original")
        code, stdout, stderr = self.run_cli(["--palier", "draft", "--graine", "42"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.strip(), str(self.out / "model_42.glb"))
        self.assertIn(english_stage("encodage de la forme"), stderr)
        self.assertNotIn("encodage de la forme", stderr)
        first, = self.history.list_runs()

        self.photo.write_bytes(b"new")
        code, _, _ = self.run_cli(["--tier", "draft", "--seed", "42", "--no-projection"])
        self.assertEqual(code, 0)
        reopened = RunHistory(self.runtime, self.out)
        latest, previous = reopened.list_runs()
        self.assertEqual(previous["id"], first["id"])
        self.assertEqual({previous["origin"], latest["origin"]}, {"cli"})
        self.assertEqual(reopened.asset(first["id"], "front").read_bytes(), b"original")
        self.assertEqual(reopened.asset(first["id"], "model").read_bytes(), GLB + b"original")
        self.assertEqual(reopened.asset(latest["id"], "model").read_bytes(), GLB + b"new")
        self.assertIsNone(reopened.asset(latest["id"], "ai"))
        self.assertIsNotNone(reopened.asset(previous["id"], "ai"))
        self.assertFalse((self.out / "model_42.ai.glb").exists())

    def test_import_pairs_legacy_assets_with_a_known_input(self):
        self.photo.write_bytes(b"original")
        model = self.out / "model_42.glb"
        model.write_bytes(GLB + b"model")
        model.with_suffix(".ai.glb").write_bytes(GLB + b"AI")
        model.with_suffix(".png").write_bytes(b"thumbnail")
        (self.out / "model_42.partiel.glb").write_bytes(GLB)

        legacy, = self.history.list_runs()
        self.assertEqual(legacy["assets"]["ai"], "model_42.ai.glb")
        self.assertIsNone(self.history.asset(legacy["id"], "front"))
        imported = self.history.import_run(model, {"front": (self.photo.name, self.photo)}, tier="draft")

        self.assertEqual(self.history.list_runs(), [imported])
        self.assertEqual(self.history.asset(imported["id"], "front").read_bytes(), b"original")
        self.assertEqual(self.history.asset(imported["id"], "thumbnail").read_bytes(), b"thumbnail")
        self.assertEqual(imported["seed"], 42)
        self.assertIsNone(self.history.asset("../out", "model"))
        self.assertIsNone(self.history.asset(imported["id"], "../run.json"))

        same = self.history.import_run(model, {"front": (self.photo.name, self.photo)}, tier="draft")
        self.assertEqual(same["id"], imported["id"])
        self.photo.write_bytes(b"corrected source")
        changed = self.history.import_run(model, {"front": (self.photo.name, self.photo)}, tier="draft")
        self.assertNotEqual(changed["id"], imported["id"])
        self.assertEqual(self.history.asset(changed["id"], "front").read_bytes(), b"corrected source")

    def test_export_error_preserves_successful_archive(self):
        self.photo.write_bytes(b"original")
        (self.out / "model_42.glb").mkdir()
        code, stdout, stderr = self.run_cli(["--tier", "draft", "--seed", "42"])
        self.assertEqual(code, 1)
        self.assertFalse(stdout)
        self.assertIn("Export failed", stderr)
        archived, = self.history.list_runs()
        self.assertEqual(archived["status"], "succeeded")
        self.assertTrue(self.history.asset(archived["id"], "model").is_file())

    def test_delete_archive_and_exact_export_preserves_original(self):
        self.photo.write_bytes(b"original")
        model = self.out / "model_42.glb"
        model.write_bytes(GLB + b"model")
        companion = model.with_suffix(".ai.glb")
        companion.write_bytes(GLB + b"ai")
        thumbnail = model.with_suffix(".png")
        thumbnail.write_bytes(b"thumbnail")
        imported = self.history.import_run(model, {"front": (self.photo.name, self.photo)})
        run_id = imported["id"]

        self.assertTrue(self.history.delete_run(run_id))
        self.assertFalse(self.history.delete_run(run_id))
        self.assertEqual(self.history.list_runs(), [])
        self.assertFalse((self.history.root / run_id).exists())
        for path in (model, companion, thumbnail):
            self.assertFalse(path.exists())
        self.assertEqual(self.photo.read_bytes(), b"original")

    def test_delete_preserves_shared_modified_and_source_files(self):
        model = self.out / "model_42.glb"
        model.write_bytes(GLB + b"model")
        companion = model.with_suffix(".ai.glb")
        companion.write_bytes(GLB + b"ai")
        thumbnail = model.with_suffix(".png")
        thumbnail.write_bytes(b"thumbnail used as an original")
        sources = {"front": (thumbnail.name, thumbnail)}
        first = self.history.import_run(model, sources, tier="draft")
        second = self.history.import_run(model, sources, tier="standard")

        self.assertTrue(self.history.delete_run(first["id"]))
        self.assertTrue(model.exists())
        self.assertTrue(companion.exists())
        companion.write_bytes(GLB + b"replacement")
        self.assertTrue(self.history.delete_run(second["id"]))
        self.assertFalse(model.exists())
        self.assertEqual(companion.read_bytes(), GLB + b"replacement")
        self.assertEqual(thumbnail.read_bytes(), b"thumbnail used as an original")

    def test_delete_preserves_newer_export_and_stale_legacy_id(self):
        self.photo.write_bytes(b"original")
        model = self.out / "model_42.glb"
        model.write_bytes(GLB + b"imported")
        companion = model.with_suffix(".ai.glb")
        companion.write_bytes(GLB + b"ai")
        imported = self.history.import_run(model, {"front": (self.photo.name, self.photo)})
        model.write_bytes(GLB + b"newer generation")
        self.assertTrue(self.history.delete_run(imported["id"]))
        self.assertEqual(model.read_bytes(), GLB + b"newer generation")
        self.assertEqual(companion.read_bytes(), GLB + b"ai")

        legacy, = self.history.list_runs()
        model.write_bytes(GLB + b"newest generation on the same path")
        self.assertFalse(self.history.delete_run(legacy["id"]))
        current, = self.history.list_runs()
        self.assertTrue(self.history.delete_run(current["id"]))
        self.assertFalse(model.exists())
        self.assertFalse(companion.exists())

    def test_delete_import_preserves_another_runs_archived_model(self):
        self.photo.write_bytes(b"original")
        code, _, _ = self.run_cli(["--tier", "draft", "--seed", "42"])
        self.assertEqual(code, 0)
        existing, = self.history.list_runs()
        model = self.history.asset(existing["id"], "model")
        imported = self.history.import_run(model, {"front": (self.photo.name, self.photo)})

        self.assertTrue(self.history.delete_run(imported["id"]))
        self.assertTrue(model.is_file())
        self.assertEqual(self.history.list_runs(), [existing])

    def test_delete_conflicts_with_cli_export_and_recovers_from_locked_file(self):
        self.photo.write_bytes(b"original")
        code, _, _ = self.run_cli(["--tier", "draft", "--seed", "42"])
        self.assertEqual(code, 0)
        completed, = self.history.list_runs()
        run_id = completed["id"]
        with self.history.cli_lock(run_id), self.assertRaises(DeletionConflict):
            self.history.delete_run(run_id)

        companion = self.out / "model_42.ai.glb"
        unlink = Path.unlink

        def blocked_unlink(path, *args, **kwargs):
            if path == companion:
                raise PermissionError("The file is open.")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", blocked_unlink), self.assertRaises(PermissionError):
            self.history.delete_run(run_id)
        self.assertEqual(self.history.get(run_id)["status"], "succeeded")
        self.assertTrue(self.history.asset(run_id, "model").exists())
        self.assertTrue(self.history.delete_run(run_id))

    def test_delete_can_retry_archive_cleanup(self):
        record = self.history.create_run(origin="web", sources={"front": ("photo.png", b"picture")})
        run_id = record["id"]
        self.history.update_run(run_id, status="failed")

        with (patch("localmesh_engine.run_history.shutil.rmtree", side_effect=PermissionError("In use")),
              self.assertRaises(PermissionError)):
            self.history.delete_run(run_id)
        self.assertIsNone(self.history.get(run_id))
        self.assertTrue(self.history.delete_run(run_id))
        self.assertFalse(self.history.delete_run(run_id))


if __name__ == "__main__":
    unittest.main()
