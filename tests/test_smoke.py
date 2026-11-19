from pathlib import Path

import yaml


def test_config_loads():
    cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    assert cfg["split"]["train_frac"] + cfg["split"]["valid_frac"] < 1
