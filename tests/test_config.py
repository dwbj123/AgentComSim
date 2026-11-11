from __future__ import annotations

import json

import pytest
import yaml

from agentcomsim.config import ExperimentConfig, load_config


def test_config_round_trip(tmp_path, small_config):
    path = tmp_path / "config.yaml"
    small_config.save_yaml(path)
    loaded = load_config(path)
    assert loaded == small_config
    json_path = tmp_path / "config.json"
    loaded.save_json(json_path)
    assert json.loads(json_path.read_text())["runs"] == 2


def test_unknown_top_level_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("runs: 2\nunknown_option: true\n")
    with pytest.raises(ValueError, match="Unknown top-level"):
        load_config(path)


def test_yaml_object_tags_are_not_executed(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text('!!python/object/apply:os.system ["echo unsafe"]\n')
    with pytest.raises(yaml.YAMLError):
        load_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 10, "warmup_s": 10},
        {"runs": 0},
        {"arrival_rates": (0.0,)},
    ],
)
def test_invalid_experiment_config(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)
