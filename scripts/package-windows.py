import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
NAME = "kubernetes-webai-rag"
PYTHON_URL = "https://www.python.org/ftp/python/3.14.4/python-3.14.4-embeddable-amd64.zip"
PYTHON_SHA256 = "fa2e63e9f0607e34d8841cc8080641069b71d2bddb5a69fa30c24eb5599da9fe"
ASSET_LIMIT = 2 * 1024**3


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def python_runtime(cache):
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "python-3.14.4-embeddable-amd64.zip"
    if destination.exists() and digest(destination) == PYTHON_SHA256:
        return destination
    temporary = destination.with_suffix(".download")
    with urllib.request.urlopen(PYTHON_URL, timeout=180) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if digest(temporary) != PYTHON_SHA256:
        raise ValueError("Official Python runtime checksum mismatch.")
    temporary.replace(destination)
    return destination


def application_files(root, release):
    paths = {root / name for name in (
        "kubernetes.html", "index.html", "OfflineVectorSearch-min.js", "NOTICE", "release.json"
    )}
    for directory in ("js", "css", "docs", "licenses", "vendor", release["embeddingModelPath"]):
        folder = root / directory
        if not folder.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid application directory: {directory}")
        members = [path for path in folder.rglob("*") if path.is_file()]
        if not members:
            raise ValueError(f"Missing application assets: {directory}")
        paths.update(members)
    manifest_path = root / "data/kubernetes/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("partial") or manifest["version"] != release["datasetVersion"]:
        raise ValueError("Release metadata must match a complete corpus.")
    paths.add(manifest_path)
    for field in ("chunks", "vectors", "keywords"):
        name = manifest[field]
        if PurePosixPath(name).name != name or "\\" in name:
            raise ValueError("Dataset filenames must stay in the data directory.")
        paths.add(manifest_path.parent / name)
    model = root / release["generationModelPath"]
    paths.add(model.parent / "README.md")
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or unsafe package input: {path}")
        if path.suffix.lower() in {".key", ".crt", ".csr", ".pem", ".pfx", ".p12", ".srl"} or path.name.endswith(".download"):
            raise ValueError(f"Temporary or credential file cannot be packaged: {path}")
    return sorted(paths)


def add_stream(archive, name, stream):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Invalid archive member: {name}")
    info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.compress_level = 1
    info.external_attr = 0o644 << 16
    with archive.open(info, "w", force_zip64=True) as output:
        shutil.copyfileobj(stream, output)


def finish_archive(temporary, destination):
    if temporary.stat().st_size >= ASSET_LIMIT:
        raise ValueError(f"Release asset must be under 2 GiB: {destination.name}")
    temporary.replace(destination)


def setup_script(version, archives):
    base = f"https://github.com/wongcyrus/{NAME}/releases/download/v{version}"
    lines = ["@echo off", "setlocal", 'cd /d "%~dp0"',
             'if not exist ".downloads" mkdir ".downloads"']
    for index, archive in enumerate(archives):
        filename = archive.name
        expected = digest(archive)
        verify = (
            '"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '
            f"\"$ErrorActionPreference='Stop'; $s=[IO.File]::OpenRead('.downloads\\{filename}'); "
            "try { $h=[Security.Cryptography.SHA256]::Create(); "
            "try { $v=[BitConverter]::ToString($h.ComputeHash($s)).Replace('-',''); "
            f"if ($v -ne '{expected}') {{ exit 1 }} "
            "} finally { $h.Dispose() } } finally { $s.Dispose() }\""
        )
        lines += [
            f'if not exist ".downloads\\{filename}" goto download_{index}',
            verify, f"if not errorlevel 1 goto ready_{index}",
            f":download_{index}",
            f'"%SystemRoot%\\System32\\curl.exe" --fail --location --retry 3 --output ".downloads\\{filename}" "{base}/{filename}"',
            "if errorlevel 1 goto failed", verify, "if errorlevel 1 goto failed",
            f":ready_{index}",
        ]
    for archive in archives:
        lines += [f'"%SystemRoot%\\System32\\tar.exe" -xf ".downloads\\{archive.name}"', "if errorlevel 1 goto failed"]
    lines += [f'echo Ready. Open "{NAME}\\Start-RAG.cmd" to run offline.',
              "pause", "exit /b 0", ":failed",
              "echo Setup failed. Check internet, disk space and the downloaded files. No server was started.",
              "pause", "exit /b 1"]
    return "\r\n".join(lines) + "\r\n"


def package(root, output, cache):
    root, output = root.resolve(), output.resolve()
    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    project = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    if release["application"] != NAME or release["version"] != project["version"]:
        raise ValueError("Run npm run prepare:offline to update the release metadata.")
    model = root / release["generationModelPath"]
    if not model.resolve().is_relative_to(root) or not model.is_file() or model.is_symlink():
        raise ValueError("Invalid model file.")
    if model.stat().st_size != release["generationModel"]["bytes"] or digest(model) != release["generationModel"]["sha256"]:
        raise ValueError("Gemma model is incomplete or has a checksum mismatch.")
    files = application_files(root, release)
    runtime = python_runtime(cache)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{NAME}-{release['version']}-windows-x64"
    app_archive = output / f"{prefix}-app.zip"
    model_archive = output / f"{prefix}-model.zip"
    temporary = app_archive.with_suffix(".zip.download")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in files:
            with path.open("rb") as stream:
                add_stream(archive, f"{NAME}/app/{path.relative_to(root).as_posix()}", stream)
        with zipfile.ZipFile(runtime) as python_zip:
            if "LICENSE.txt" not in python_zip.namelist():
                raise ValueError("Python runtime license is missing.")
            for entry in sorted(python_zip.namelist()):
                if not entry.endswith("/"):
                    with python_zip.open(entry) as stream:
                        add_stream(archive, f"{NAME}/runtime/{entry}", stream)
        for name in ("serve.py", "Start-RAG.cmd"):
            with (ROOT / "windows" / name).open("rb") as stream:
                add_stream(archive, f"{NAME}/{name}", stream)
    finish_archive(temporary, app_archive)
    temporary = model_archive.with_suffix(".zip.download")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive, model.open("rb") as stream:
        add_stream(archive, f"{NAME}/app/{model.relative_to(root).as_posix()}", stream)
    finish_archive(temporary, model_archive)
    archives = [app_archive, model_archive]
    with (output / "Setup-RAG.cmd").open("w", encoding="ascii", newline="") as stream:
        stream.write(setup_script(release["version"], archives))
    (output / "SHA256SUMS.txt").write_text("".join(
        f"{digest(path)}  {path.name}\n" for path in [*archives, output / "Setup-RAG.cmd"]
    ), encoding="ascii")
    for path in [*archives, output / "Setup-RAG.cmd", output / "SHA256SUMS.txt"]:
        print(f"{path.name}: {path.stat().st_size:,} bytes")
    print("Prepared release assets only; nothing was committed or published.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package a complete offline site for Windows 11 x64.")
    parser.add_argument("--site", type=Path, default=ROOT / "offline")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    package(args.site, args.out, args.cache)
