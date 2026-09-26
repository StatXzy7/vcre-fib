from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_bytes(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(value)
    os.replace(tmp, path)


def save_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    torch.save(value, tmp)
    os.replace(tmp, path)


def source_fingerprint():
    import sfibai_b
    roots = [Path(__file__).parent, Path(sfibai_b.__file__).parent]
    result = {root.name + "/" + p.name: sha256(p) for root in roots for p in sorted(root.glob("*.py"))}
    package = Path(__file__).parents[2]
    result.update({"protocol/" + n: sha256(package / n) for n in ("PROTOCOL.md", "SEARCH_SPACE.yaml", "pyproject.toml")})
    return result


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "pandas", "optuna", "PyYAML", "opencv-python", "scikit-learn")},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def cache_key(source_hash, annotation_hash, recipe):
    return digest({"source": source_hash, "annotations": annotation_hash, "recipe": recipe})


class RoundLock:
    """OS-level lock released on process death; prevents competing controllers."""
    def __init__(self, output):
        self.path = Path(output) / "controller.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another controller is active") from None
        return self

    def __exit__(self, *_):
        self.handle.close()
