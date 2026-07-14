from video2humanoid.config import load_config
from video2humanoid.registry import get_extractor, get_retargeter, get_simulator


def test_registry_backends():
    assert get_extractor("gemx") is not None
    assert get_retargeter("soma_g1") is not None
    assert get_simulator("isaac_kinematic") is not None


def test_load_config():
    config = load_config("configs/config.yaml")
    assert config.extractor["name"] == "gemx"
