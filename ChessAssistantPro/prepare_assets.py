"""Stage ONNX Runtime for linking and Maia ONNX for a rootless package."""
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".asset-cache"
MODEL = ROOT / "layout/Library/Application Support/ChessAssistantPro/maia3_79m_int32.onnx"
FRAMEWORK = CACHE / "ort/ios-arm64/onnxruntime.framework"
ORT_URL = "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.24.2.zip"
ORT_SHA256 = "f7100a992d2a8135168c8afd831e6a58b465349101982aa58b3e11d36e600b54"
HF_API = "https://huggingface.co/api/models/cstr/maia3-onnx-int32/tree/main?recursive=true"
MODEL_URL = "https://huggingface.co/cstr/maia3-onnx-int32/resolve/main/maia3_79m_int32.onnx"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download(url, expected, destination):
    if destination.is_file() and sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            pending = Path(output.name)
            request = urllib.request.Request(url, headers={"User-Agent": "ChessAssistantPro/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        actual = sha256(pending)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {url}: {actual}")
        pending.replace(destination)
    finally:
        if pending:
            pending.unlink(missing_ok=True)

def model_sha256():
    request = urllib.request.Request(HF_API, headers={"User-Agent": "ChessAssistantPro/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.load(response)
    for item in entries:
        if item.get("path") == "maia3_79m_int32.onnx":
            checksum = item.get("lfs", {}).get("oid", "").removeprefix("sha256:")
            if len(checksum) == 64:
                return checksum
    raise RuntimeError("Hugging Face did not publish the Maia ONNX LFS checksum")

def stage_runtime():
    if (FRAMEWORK / "onnxruntime").is_file() and (FRAMEWORK / "Headers/onnxruntime_c_api.h").is_file():
        return
    archive = CACHE / "onnxruntime-c-1.24.2.zip"
    download(ORT_URL, ORT_SHA256, archive)
    prefix = "onnxruntime.xcframework/ios-arm64/onnxruntime.framework/"
    with zipfile.ZipFile(archive) as source:
        matches = [entry for entry in source.infolist() if prefix in entry.filename and not entry.is_dir()]
        if not matches:
            raise RuntimeError("Official ONNX Runtime archive has no iOS ARM64 framework")
        for entry in matches:
            suffix = entry.filename.split(prefix, 1)[1]
            if ".." in Path(suffix).parts:
                raise RuntimeError("Unsafe framework path")
            target = FRAMEWORK / suffix
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
    if not (FRAMEWORK / "onnxruntime").is_file() or not (FRAMEWORK / "Headers/onnxruntime_c_api.h").is_file():
        raise RuntimeError("ONNX Runtime library or C header is missing")

def main():
    stage_runtime()
    expected = os.environ.get("MAIA79_ONNX_SHA256") or model_sha256()
    if len(expected) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected):
        raise RuntimeError("MAIA79_ONNX_SHA256 must be a SHA-256 hex digest")
    download(MODEL_URL, expected.lower(), MODEL)
    if MODEL.stat().st_size < 200_000_000:
        raise RuntimeError("Maia 79M ONNX file is unexpectedly small")
    print(f"Staged {MODEL} and {FRAMEWORK}")

if __name__ == "__main__":
    main()
