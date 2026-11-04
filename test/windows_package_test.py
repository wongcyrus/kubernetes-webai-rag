import functools
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load_module("rag_server", "windows/serve.py")
packager = load_module("rag_packager", "scripts/package-windows.py")


class WindowsPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "app"
        self.root.mkdir()
        version = json.loads((ROOT / "package.json").read_text())["version"]
        self.release = {
            "application": "kubernetes-webai-rag", "version": version, "datasetVersion": "test-version",
            "generationModelPath": "models/gemma/model.litertlm",
            "generationModel": {"bytes": 2, "sha256": ""},
            "embeddingModelPath": "models/embedding",
        }
        for name in ("kubernetes.html", "index.html", "OfflineVectorSearch-min.js", "NOTICE",
                     "models/gemma/README.md", "models/gemma/model.litertlm",
                     "models/embedding/model.onnx", "vendor/runtime.wasm",
                     "js/kubernetes.js", "css/style.css", "docs/index.html", "licenses/project.txt"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ok")
        self.release["generationModel"]["sha256"] = packager.digest(self.root / self.release["generationModelPath"])
        data = self.root / "data/kubernetes"
        data.mkdir(parents=True)
        self.manifest = {
            "version": "test-version", "partial": False,
            "chunks": "chunks.json", "vectors": "vectors.f32", "keywords": "keywords.json",
        }
        (data / "manifest.json").write_text(json.dumps(self.manifest))
        for name in ("chunks.json", "vectors.f32", "keywords.json"):
            (data / name).write_bytes(b"ok")
        self.write_release()

    def write_release(self):
        (self.root / "release.json").write_text(json.dumps(self.release))

    def test_server_rejects_missing_partial_or_outside_model(self):
        server.check_package(self.root)
        (self.root / self.release["generationModelPath"]).write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "incomplete Gemma"):
            server.check_package(self.root)
        self.release["generationModelPath"] = "../outside.litertlm"
        self.write_release()
        with self.assertRaisesRegex(ValueError, "Invalid model path"):
            server.check_package(self.root)

    def test_server_serves_runtime_mime_types_health_and_revalidates_ui(self):
        handler = functools.partial(server.Handler, directory=str(self.root), release=self.release)
        instance = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=instance.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{instance.server_port}"
            with urllib.request.urlopen(base + "/healthz") as response:
                self.assertEqual(json.load(response)["application"], "kubernetes-webai-rag")
            for name, mime in (("OfflineVectorSearch-min.js", "text/javascript"), ("vendor/runtime.wasm", "application/wasm")):
                with urllib.request.urlopen(base + "/" + name) as response:
                    self.assertEqual(response.headers["Content-Type"], mime)
                    self.assertEqual(response.headers["Cache-Control"], "no-cache")
                    self.assertEqual(response.read(), b"ok")
        finally:
            instance.shutdown()
            instance.server_close()
            thread.join()

    def test_release_separates_model_and_excludes_development_files(self):
        (self.root / "local-test-ca.crt").write_text("public test certificate")
        (self.root / "private.key").write_text("not a real key")
        (self.root / "data/kubernetes/old-vectors.f32").write_bytes(b"old")
        runtime = Path(self.temporary.name) / "python.zip"
        with zipfile.ZipFile(runtime, "w") as archive:
            archive.writestr("LICENSE.txt", "test license")
            archive.writestr("python.exe", b"test runtime")
        output = Path(self.temporary.name) / "dist"
        with patch.object(packager, "python_runtime", return_value=runtime):
            packager.package(self.root, output, Path(self.temporary.name) / "cache")
        app = next(output.glob("*-app.zip"))
        model = next(output.glob("*-model.zip"))
        with zipfile.ZipFile(app) as archive:
            names = archive.namelist()
            self.assertIn("kubernetes-webai-rag/runtime/LICENSE.txt", names)
            self.assertIn("kubernetes-webai-rag/app/vendor/runtime.wasm", names)
            self.assertIn("kubernetes-webai-rag/Start-RAG.cmd", names)
            self.assertFalse(any(name.endswith((".crt", ".key", ".litertlm", "old-vectors.f32")) for name in names))
        with zipfile.ZipFile(model) as archive:
            self.assertEqual(archive.namelist(), ["kubernetes-webai-rag/app/models/gemma/model.litertlm"])
            self.assertEqual(archive.read(archive.namelist()[0]), b"ok")
        script = (output / "Setup-RAG.cmd").read_text()
        self.assertIn(packager.digest(app), script)
        self.assertIn(packager.digest(model), script)
        self.assertIn("[Security.Cryptography.SHA256]::Create()", script)
        self.assertIn("ComputeHash($s)", script)
        self.assertNotIn("ExecutionPolicy", script)
        self.assertIn("if errorlevel 1 goto failed", script)
        if os.name == "nt":
            downloads = output / ".downloads"
            downloads.mkdir()
            for archive in (app, model):
                shutil.copyfile(archive, downloads / archive.name)
            subprocess.run(["cmd.exe", "/d", "/c", "Setup-RAG.cmd"],
                           cwd=output, input="\r\n", text=True, timeout=30, check=True)
            server.check_package(output / "kubernetes-webai-rag/app")

    def test_server_suppresses_disconnected_downloads_and_keeps_serving(self):
        handler = functools.partial(server.Handler, directory=str(self.root), release=self.release)
        instance = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=instance.serve_forever)
        with patch.object(instance, "handle_error") as handle_error, patch.object(
            server.Handler, "copyfile", side_effect=[ConnectionResetError(), BrokenPipeError()]
        ):
            thread.start()
            try:
                base = f"http://127.0.0.1:{instance.server_port}"
                for _ in range(2):
                    with urllib.request.urlopen(base + "/vendor/runtime.wasm", timeout=5) as response:
                        with self.assertRaises(IncompleteRead):
                            response.read()
                with urllib.request.urlopen(base + "/healthz", timeout=5) as response:
                    self.assertEqual(json.load(response)["application"], "kubernetes-webai-rag")
                handle_error.assert_not_called()
            finally:
                instance.shutdown()
                instance.server_close()
                thread.join()

    def test_server_does_not_hide_unexpected_errors(self):
        handler = object.__new__(server.Handler)
        with patch.object(server.SimpleHTTPRequestHandler, "handle", side_effect=OSError("unexpected failure")):
            with self.assertRaisesRegex(OSError, "unexpected failure"):
                handler.handle()

    def test_packager_refuses_changed_model_partial_data_and_oversized_assets(self):
        self.manifest["partial"] = True
        (self.root / "data/kubernetes/manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "complete corpus"):
            packager.application_files(self.root, self.release)
        (self.root / self.release["generationModelPath"]).write_bytes(b"xx")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            packager.package(self.root, Path(self.temporary.name) / "dist", Path(self.temporary.name) / "cache")
        temporary = Path(self.temporary.name) / "oversized.download"
        temporary.write_bytes(b"1234")
        destination = temporary.with_suffix(".zip")
        with patch.object(packager, "ASSET_LIMIT", 4):
            with self.assertRaisesRegex(ValueError, "under 2 GiB"):
                packager.finish_archive(temporary, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
