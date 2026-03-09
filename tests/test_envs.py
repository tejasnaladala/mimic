import pytest

from mimic.config.models import CameraConfig, EnvConfig, TeleopConfig, TrainConfig
from mimic.envs.registry import list_envs


class TestConfigs:
    def test_env_config_defaults(self):
        config = EnvConfig(name="test")
        assert config.robot == "panda"
        assert config.control_hz == 20
        assert config.physics_hz == 1000
        assert len(config.cameras) == 2
        assert config.cameras[0].name == "front"
        assert config.cameras[1].name == "wrist"
        assert config.episode_length == 300
        assert config.action_space == "joint"

    def test_env_config_custom(self):
        config = EnvConfig(
            name="custom",
            robot="ur5e",
            control_hz=50,
            cameras=[CameraConfig(name="overhead", width=640, height=480)],
        )
        assert config.robot == "ur5e"
        assert config.control_hz == 50
        assert len(config.cameras) == 1

    def test_camera_config_defaults(self):
        cam = CameraConfig(name="test_cam")
        assert cam.width == 320
        assert cam.height == 240
        assert cam.fovy == 45.0

    def test_train_config_defaults(self):
        config = TrainConfig()
        assert config.policy == "act"
        assert config.batch_size == 32
        assert config.lr == 1e-4

    def test_teleop_config_defaults(self):
        config = TeleopConfig()
        assert config.port == 8765
        assert config.control_mode == "cartesian"


class TestRegistry:
    def test_list_envs_returns_list(self):
        result = list_envs()
        assert isinstance(result, list)

    def test_make_unknown_env_raises(self):
        with pytest.raises(ValueError, match="Unknown env"):
            from mimic.envs.registry import make

            make("nonexistent_env_12345")
