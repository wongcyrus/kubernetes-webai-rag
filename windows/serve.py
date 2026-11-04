import argparse
import functools
import json
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


APPLICATION = "kubernetes-webai-rag"
URL = "http://localhost:8080/kubernetes.html"


def check_package(root):
    root = Path(root).resolve()
    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    if release["application"] != APPLICATION:
        raise ValueError("This is not a Kubernetes WebAI RAG package.")
    for name in ("kubernetes.html", "OfflineVectorSearch-min.js", "data/kubernetes/manifest.json"):
        if not (root / name).is_file():
            raise ValueError(f"Missing application file: {name}")
    model = (root / release["generationModelPath"]).resolve()
    if not model.is_relative_to(root):
        raise ValueError("Invalid model path in release metadata.")
    if not model.is_file() or model.stat().st_size != release["generationModel"]["bytes"]:
        raise ValueError("Missing or incomplete Gemma model. Extract the model archive into the same folder.")
    return {key: release[key] for key in ("application", "version", "datasetVersion")}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript", ".mjs": "text/javascript",
        ".wasm": "application/wasm", ".f32": "application/octet-stream",
        ".onnx": "application/octet-stream", ".litertlm": "application/octet-stream",
    }

    def __init__(self, *args, release=None, **kwargs):
        self.release = release
        super().__init__(*args, **kwargs)

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def do_GET(self):
        if urlsplit(self.path).path == "/healthz":
            body = json.dumps(self.release).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "public, max-age=31536000, immutable"
                         if urlsplit(self.path).path.startswith("/models/") else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404, "Directory listing is disabled.")


def run(root, open_browser=True):
    try:
        release = check_package(root)
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot start: {error}")
        return 1
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/healthz", timeout=1) as response:
            if json.load(response) == release:
                print(f"Already running: {URL}")
                if open_browser:
                    webbrowser.open(URL)
                return 0
    except (OSError, ValueError):
        pass
    handler = functools.partial(Handler, directory=str(root), release=release)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 8080), handler)
    except OSError as error:
        print(f"Cannot use localhost port 8080: {error}")
        print("Close the other server and retry. The port is fixed to preserve the browser model cache.")
        return 1
    print(f"Kubernetes WebAI RAG v{release['version']}: {URL}\nKeep this window open. Press Ctrl+C to stop.")
    if open_browser:
        timer = threading.Timer(0.5, webbrowser.open, args=(URL,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the local server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve Kubernetes WebAI RAG on localhost.")
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "app")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.directory.resolve(), not args.no_browser))
