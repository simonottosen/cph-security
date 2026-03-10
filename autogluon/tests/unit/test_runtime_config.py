from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _import_app_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("project.app")


def test_detect_default_device_map_falls_back_to_cpu_when_torch_missing(monkeypatch):
    app = _import_app_module()

    def _raise(_name):
        raise ImportError("torch unavailable")

    monkeypatch.setattr(app.platform, "system", lambda: "Linux")
    monkeypatch.setattr(app.importlib, "import_module", _raise)

    assert app._detect_default_device_map() == "cpu"


def test_detect_default_device_map_uses_cuda_when_available(monkeypatch):
    app = _import_app_module()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )

    monkeypatch.setattr(app.platform, "system", lambda: "Linux")
    monkeypatch.setattr(app.importlib, "import_module", lambda _name: fake_torch)

    assert app._detect_default_device_map() == "cuda"


def test_detect_default_device_map_uses_cpu_when_cuda_unavailable(monkeypatch):
    app = _import_app_module()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )

    monkeypatch.setattr(app.platform, "system", lambda: "Linux")
    monkeypatch.setattr(app.importlib, "import_module", lambda _name: fake_torch)

    assert app._detect_default_device_map() == "cpu"
