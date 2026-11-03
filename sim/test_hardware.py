"""Hardware boundary tests using in-memory messages and a fake robot state."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pylibfranka import JointPositions

from common import (HOME, STATE, TARGET, command, joint_ctrl_t, joint_meas_t,
                    pose3d_t, load_model, drain)
from run_controller import Controller
from run_hardware import (MockRobot, StartMove, VelocityGuard, move_to_start, run,
                          MAX_SPEED, MAX_ACCEL, MAX_JERK, TIMEOUT)
from run_mirror import Mirror

# Inside the joint limits, but under the virtual cell's floor margin.
NEAR_FLOOR = np.array([0., 1.4, 0., -.5, 0., 1., 0.])


class HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = load_model()

    def state(self):
        return SimpleNamespace(q=HOME.copy(), dq_d=np.zeros(7), ddq_d=np.zeros(7))

    def packet(self, dq=None, stamp=1.):
        if dq is None:
            dq = np.ones(7) * .1
        return joint_ctrl_t.decode(command(HOME.copy(), np.array(dq), stamp))

    def test_packet_validation(self):
        guard, state = VelocityGuard(self.model.jnt_range), self.state()
        for msg in [self.packet(np.ones(7) * np.nan), self.packet(np.ones(7)),
                    self.packet(stamp=.5), self.packet(stamp=1.5)]:
            self.assertFalse(guard.accept(10., msg, state, 1., 10.))
        self.assertFalse(guard.accept(9., self.packet(), state, 1., 10.))
        wrong_position = self.packet()
        wrong_position.q_des = list(wrong_position.q_des)
        wrong_position.q_des[0] += .5
        self.assertFalse(guard.accept(10., wrong_position, state, 1., 10.))
        invalid = self.packet()
        invalid.valid = False
        self.assertFalse(guard.accept(10., invalid, state, 1., 10.))
        self.assertTrue(guard.accept(10., self.packet(), state, 1., 10.))
        self.assertFalse(guard.accept(10., self.packet(), state, 1., 10.))

    def test_velocity_acceleration_jerk_bounds_and_stop(self):
        guard, state = VelocityGuard(self.model.jnt_range), self.state()
        self.assertTrue(guard.accept(10., self.packet(), state, 1., 10.))
        previous_acc = np.zeros(7)
        previous_v = np.zeros(7)
        distance = np.zeros(7)
        for tick in range(3000):
            # Exercise forward motion, reversal and controlled shutdown.
            guard.last_receive = 10. + tick * .001
            guard.desired[:] = .1 if tick < 800 else -.1
            v = guard.velocity(state, .001, guard.last_receive, stopping=tick >= 1600)
            a = (v - previous_v) / .001
            self.assertLessEqual(np.max(np.abs(v)), MAX_SPEED + 1e-8)
            self.assertLessEqual(np.max(np.abs(a)), MAX_ACCEL + 1e-8)
            self.assertLessEqual(np.max(np.abs(a - previous_acc)) / .001, MAX_JERK + 1e-7)
            state.dq_d, state.ddq_d = v, a
            state.q = np.array(state.q) + v * .001
            distance += v * .001
            previous_v, previous_acc = v, a
        self.assertLess(np.max(np.abs(previous_v)), 1e-5)
        self.assertLess(np.max(np.abs(previous_acc)), .01)

    def test_expired_command_brakes(self):
        guard, state = VelocityGuard(self.model.jnt_range), self.state()
        guard.accept(10., self.packet(), state, 1., 10.)
        # With an expired packet, an initially stationary robot stays stationary.
        np.testing.assert_array_equal(guard.velocity(state, .001, 10. + TIMEOUT + .01), np.zeros(7))
        state.dq_d = np.ones(7) * .05
        v = guard.velocity(state, .001, 11.)
        self.assertTrue(np.all(v < .05))

    def test_bad_cycle_and_joint_margin_fault(self):
        guard, state = VelocityGuard(self.model.jnt_range), self.state()
        for dt in (0., .006, np.nan):
            with self.assertRaises(RuntimeError):
                guard.velocity(state, dt, 1.)
        state.q[0] = self.model.jnt_range[0, 1]
        with self.assertRaises(RuntimeError):
            guard.velocity(state, .001, 1.)

    def test_mirror_uses_measured_joints_without_integration(self):
        mirror = Mirror("memq://")
        q = HOME + .01
        msg = joint_meas_t()
        msg.timestamp, msg.num_joints = 1., 7
        msg.q, msg.qd, msg.tau = q.tolist(), [.1] * 7, [0.] * 7
        mirror.bus.publish(STATE, msg.encode())
        drain(mirror.bus)
        np.testing.assert_allclose(mirror.data.qpos, q, atol=1e-7)
        self.assertEqual(mirror.data.time, 0.)
        self.assertTrue(mirror.have_state)
        target = pose3d_t()
        target.position = [.4, .1, .7]
        mirror.bus.publish(TARGET, target.encode())
        drain(mirror.bus)
        np.testing.assert_allclose(mirror.data.mocap_pos[0], target.position, atol=1e-7)
        np.testing.assert_allclose(mirror.data.qpos, q, atol=1e-7)

    def start_move(self, q, stop_after=np.inf):
        move, path = StartMove(np.array(q)), []
        # libfranka's first period is zero.
        q, finished = move.step(0., False)
        path.append(q)
        while not finished:
            q, finished = move.step(.001, len(path) * .001 >= stop_after)
            path.append(q)
            self.assertLess(len(path), 20000)
        return np.array(path)

    def rates(self, path):
        v = np.diff(path, axis=0) / .001
        return np.max(np.abs(v)), np.max(np.abs(np.diff(v, axis=0) / .001)), v

    def test_start_move_reaches_pose_at_twenty_percent_speed(self):
        path = self.start_move(HOME)
        np.testing.assert_array_equal(path[0], HOME)
        np.testing.assert_allclose(path[-1], [0, 0, 0, -1.5707963, 0, 1.5707963, 0], atol=1e-6)
        speed, acceleration, _ = self.rates(path)
        self.assertLessEqual(speed, .4 + 1e-9)
        self.assertLessEqual(acceleration, 1.)
        # Joint 7 travels .75 rad, so a .4 rad/s peak needs about 3.5 s.
        self.assertAlmostEqual(len(path) * .001, 3.5, delta=.1)

    def test_start_move_keeps_short_moves_gentle(self):
        speed, acceleration, _ = self.rates(self.start_move(
            [.1, 0, 0, -1.5707963, 0, 1.5707963, 0]))
        self.assertLessEqual(speed, .4)
        self.assertLessEqual(acceleration, 1.)

    def test_start_move_eases_to_rest_when_stopped_at_peak_speed(self):
        path = self.start_move(HOME, stop_after=1.75)
        speed, acceleration, v = self.rates(path)
        self.assertLessEqual(acceleration, 1.)
        self.assertLess(np.max(np.abs(np.diff(v, 2, axis=0) / .001 ** 2)), 50.)
        self.assertLess(np.max(np.abs(v[-1])), 1e-3)
        # It stops well short of joint 7 = 0, inside a second of the request.
        self.assertGreater(path[-1, 6], .2)
        self.assertLess(len(path) * .001, 2.75)

    def test_move_to_start_finishes_so_velocity_control_can_begin(self):
        robot, sent = MockRobot(), []
        robot.state.q_d = (HOME + .001).tolist()
        active = robot.start_joint_position_control(None)

        def positions(q):
            sent.append(q)
            return JointPositions(q)

        with patch("run_hardware.time.sleep"):
            move_to_start(active, positions, lambda: False)
        np.testing.assert_allclose(sent[0], HOME + .001, atol=1e-9)
        np.testing.assert_allclose(robot.state.q, [0, 0, 0, -1.5707963, 0, 1.5707963, 0], atol=1e-6)
        robot.start_joint_velocity_control(None)

    def test_mock_refuses_a_second_motion_while_one_is_running(self):
        robot = MockRobot()
        robot.start_joint_position_control(None)
        with self.assertRaises(RuntimeError):
            robot.start_joint_velocity_control(None)

    def run_mock(self, robot, move_to_start=True):
        args = SimpleNamespace(mock=True, control=True, move_to_start=move_to_start, ip="",
                               url="memq://", seconds=.05)
        with patch("run_hardware.MockRobot", return_value=robot), patch("run_hardware.time.sleep"):
            run(args)

    def test_start_move_may_begin_outside_cell(self):
        robot = MockRobot()
        robot.state.q = robot.state.q_d = NEAR_FLOOR.tolist()
        self.run_mock(robot)
        np.testing.assert_allclose(robot.state.q, [0, 0, 0, -1.5707963, 0, 1.5707963, 0], atol=1e-6)

    def test_control_refuses_pose_outside_cell_after_start_move(self):
        robot = MockRobot()
        with patch("run_hardware.START", NEAR_FLOOR), self.assertRaisesRegex(RuntimeError, "clearance"):
            self.run_mock(robot)
        np.testing.assert_allclose(robot.state.q, NEAR_FLOOR, atol=1e-6)
        self.assertFalse(robot.active)

    def test_control_refuses_pose_outside_cell_without_start_move(self):
        robot = MockRobot()
        robot.state.q = robot.state.q_d = NEAR_FLOOR.tolist()
        with self.assertRaisesRegex(RuntimeError, "clearance"):
            self.run_mock(robot, move_to_start=False)
        self.assertFalse(robot.active)

    def test_controller_drops_old_target_when_adapter_restarts(self):
        def measured(q, stamp):
            msg = joint_meas_t()
            msg.timestamp, msg.num_joints = stamp, 7
            msg.q, msg.qd, msg.tau = q.tolist(), [0.] * 7, [0.] * 7
            return msg.encode()
        ctrl = Controller("memq://", hardware=True)
        ctrl.on_state(STATE, measured(HOME, 5.))
        ctrl.target += [.1, 0, 0]
        ctrl.on_state(STATE, measured(HOME + .1, .001))
        np.testing.assert_allclose(ctrl.target, ctrl.ik.tip()[0], atol=1e-9)

    def test_mirror_target_shows_through_arm(self):
        model = Mirror("memq://").model
        arm = model.geom_matid[model.geom_group == 2]
        self.assertTrue(np.all(model.mat_rgba[arm, 3] < 1))
        self.assertEqual(model.geom_rgba[model.body("keyboard_target").geomadr[0], 3], 1)


if __name__ == "__main__":
    unittest.main()
