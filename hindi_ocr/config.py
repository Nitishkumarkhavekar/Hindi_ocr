"""Configuration: hindi_ocr.yaml next to the package (or a path / dict passed in). Paths resolve relative to the yaml's folder."""
import os
from pathlib import Path
import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = PKG_ROOT / "hindi_ocr.yaml"


class Config:
    def __init__(self, source=None, **overrides):
        if isinstance(source, dict):
            self.root, self.data = PKG_ROOT, dict(source)
        else:
            path = Path(source) if source else DEFAULT_YAML
            self.root = path.resolve().parent
            with open(path, encoding="utf-8") as f:
                self.data = yaml.safe_load(f) or {}
        for key, value in overrides.items():       # "detector.thresh": 0.2 style overrides
            section, _, name = key.partition(".")
            if name:
                self.data.setdefault(section, {})[name] = value
            else:
                self.data[section] = value

    def __getitem__(self, section):
        return self.data.get(section, {})

    def path(self, section, key):
        p = Path(self[section][key])
        return p if p.is_absolute() else self.root / p
