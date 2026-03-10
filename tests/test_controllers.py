import mimic.envs.tasks  # noqa: F401
from mimic.envs.registry import make
from mimic.teleop.commands import CommandRouter
from mimic.teleop.controllers.cartesian import CartesianController
from mimic.teleop.controllers.joint import JointController


class TestJointController:
    def test_creates(self):
        env = make("pick-place")
        ctrl = JointController(env)
        assert ctrl.env is env
        env.close()

    def test_joint_delta(self):
        env = make("pick-place")
        env.reset()
        ctrl = JointController(env, speed=1.0)
        action = ctrl.process_command({"type": "joint_delta", "joint": 0, "delta": 0.5})
        assert action is not None
        assert action.shape == (env.action_dim,)
        env.close()

    def test_gripper_command(self):
        env = make("pick-place")
        env.reset()
        ctrl = JointController(env)
        action = ctrl.process_command({"type": "gripper", "value": 1.0})
        assert action is not None
        # Gripper should be open (positive value)
        if env.action_dim > 7:
            assert action[7] > 0
        env.close()

    def test_reset_command(self):
        env = make("pick-place")
        env.reset()
        ctrl = JointController(env)
        result = ctrl.process_command({"type": "reset"})
        assert result is None  # signals env reset
        env.close()


class TestCartesianController:
    def test_creates(self):
        env = make("pick-place")
        ctrl = CartesianController(env)
        assert ctrl.env is env
        env.close()

    def test_cartesian_delta(self):
        env = make("pick-place")
        env.reset()
        ctrl = CartesianController(env)
        action = ctrl.process_command({
            "type": "cartesian_delta",
            "dx": 1.0,
            "dy": 0.0,
            "dz": 0.0,
        })
        assert action is not None
        assert action.shape == (env.action_dim,)
        env.close()

    def test_get_ee_pos(self):
        env = make("pick-place")
        env.reset()
        ctrl = CartesianController(env)
        pos = ctrl.get_ee_pos()
        assert pos.shape == (3,)
        env.close()


class TestCommandRouter:
    def test_creates(self):
        env = make("pick-place")
        router = CommandRouter(env, mode="joint")
        assert router.mode == "joint"
        env.close()

    def test_mode_switch(self):
        env = make("pick-place")
        env.reset()
        router = CommandRouter(env)
        resp = router.process({"type": "set_mode", "mode": "cartesian"})
        assert resp["mode"] == "cartesian"
        env.close()

    def test_reset_command(self):
        env = make("pick-place")
        env.reset()
        router = CommandRouter(env)
        resp = router.process({"type": "reset"})
        assert resp["status"] == "ok"
        env.close()

    def test_joint_command_updates_target(self):
        env = make("pick-place")
        env.reset()
        router = CommandRouter(env)
        resp = router.process({"type": "joint_delta", "joint": 0, "delta": 0.1})
        assert resp["status"] == "ok"
        env.close()

    def test_recording_toggle(self):
        env = make("pick-place")
        env.reset()
        router = CommandRouter(env)
        resp = router.process({"type": "start_recording"})
        assert resp["recording"] is True
        resp = router.process({"type": "stop_recording"})
        assert resp["recording"] is False
        env.close()

    def test_unknown_command(self):
        env = make("pick-place")
        env.reset()
        router = CommandRouter(env)
        resp = router.process({"type": "nonexistent"})
        assert resp["status"] == "unknown_command"
        env.close()
