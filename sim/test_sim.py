"""Offline checks: model mapping, wire compatibility, QP and dynamic wall approach."""
import unittest
import time

import mujoco
import numpy as np

from common import (HOME, DT, MARGIN, SPEED, STATE, COMMAND, ROBOT, STATIC,
                    load_model, collision_geometry, encode_geometry,
                    decode_geometry, sphere_box, command, drain, IK)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = load_model()

    def make_ik(self):
        ik = IK(self.model)
        robot, walls = collision_geometry(self.model)
        # Exercise the actual wire format at the controller boundary.
        ik.robot = decode_geometry(encode_geometry(robot, 0.))
        ik.walls = decode_geometry(encode_geometry(walls, 0.))
        ik.set_state(HOME, np.zeros(7))
        return ik

    def test_mapping_and_jacobian(self):
        m = self.model
        self.assertEqual((m.nq, m.nv, m.nu), (7, 7, 7))
        for i in range(7):
            self.assertEqual(m.joint(i).name, f"fr3_joint{i+1}")
            self.assertEqual(int(m.actuator_trnid[i, 0]), i)
        ik = self.make_ik()
        p, J = ik.tip()
        for i in range(7):
            q = HOME.copy()
            q[i] += 1e-6
            ik.set_state(q, np.zeros(7))
            np.testing.assert_allclose((ik.tip()[0] - p) / 1e-6, J[:, i], atol=1e-6)

    def test_geometry(self):
        ik = self.make_ik()
        self.assertEqual(len(ik.robot), 8)
        self.assertEqual(len(set(x[0] for x in ik.robot)), 8)
        self.assertEqual(len(ik.walls), 5)
        self.assertGreater(min(d for d, _ in ik.collision_rows()), MARGIN)
        distance, normal = sphere_box(np.array([2., 0, 0]), .1, np.ones(3)*2, np.eye(4))
        self.assertAlmostEqual(distance, .9)
        np.testing.assert_allclose(normal, [1, 0, 0])
        distance, normal = sphere_box(np.array([.8, 0, 0]), .1, np.ones(3)*2, np.eye(4))
        self.assertAlmostEqual(distance, -.3)
        np.testing.assert_allclose(normal, [1, 0, 0])

    def rollout(self, offset, seconds):
        ik = self.make_ik()
        d = mujoco.MjData(self.model)
        d.qpos[:] = HOME
        d.ctrl[:] = HOME
        mujoco.mj_forward(self.model, d)
        target = ik.tip()[0] + offset
        q_des = HOME.copy()
        minimum = float("inf")
        for _ in range(round(seconds / DT)):
            ik.set_state(d.qpos, d.qvel)
            dq = ik.solve(target)
            self.assertTrue(np.isfinite(dq).all())
            self.assertLessEqual(np.max(np.abs(dq)), SPEED + 2e-5)
            q_des += DT * dq
            q_des = np.clip(q_des, d.qpos - .03, d.qpos + .03)
            for _ in range(5):
                d.ctrl[:] = q_des
                d.qfrc_applied[:] = d.qfrc_bias
                mujoco.mj_step(self.model, d)
            ik.set_state(d.qpos, d.qvel)
            minimum = min(minimum, min(dist for dist, _ in ik.collision_rows()))
            self.assertTrue(np.isfinite(d.qpos).all())
            self.assertTrue(np.all(d.qpos >= self.model.jnt_range[:, 0]))
            self.assertTrue(np.all(d.qpos <= self.model.jnt_range[:, 1]))
        return np.linalg.norm(target - ik.tip()[0]), minimum, ik

    def test_target_tracking(self):
        error, clearance, _ = self.rollout(np.array([.02, .08, -.02]), 4.)
        self.assertLess(error, .01)
        self.assertGreater(clearance, MARGIN - .005)

    def test_unreachable_target_beyond_wall(self):
        error, clearance, _ = self.rollout(np.array([.8, 0, 0]), 8.)
        self.assertGreater(error, .3)
        self.assertGreater(clearance, MARGIN - .005)

    def test_missing_geometry_stops(self):
        ik = IK(self.model)
        ik.set_state(HOME, np.zeros(7))
        np.testing.assert_array_equal(ik.solve(np.zeros(3)), np.zeros(7))

    def test_infeasible_qp_stops(self):
        ik = self.make_ik()
        # A zero Jacobian cannot satisfy a demanded gap-opening velocity.
        ik.collision_rows = lambda: iter([(-1., np.zeros(7))])
        np.testing.assert_array_equal(ik.solve(np.zeros(3)), np.zeros(7))
        self.assertIn("infeasible", ik.status)

    def test_command_validation_and_watchdog(self):
        from run_env import Environment
        env = Environment("memq://")
        # Bad numbers, excessive speed and old simulator timestamps are rejected.
        for q, dq, stamp in [(HOME * np.nan, np.zeros(7), 0),
                              (HOME, np.ones(7), 0),
                              (HOME, np.zeros(7), -1)]:
            env.on_command(COMMAND, command(q, dq, stamp))
            self.assertTrue(env.expired)
        env.on_command(COMMAND, command(HOME + .001, np.zeros(7), 0))
        self.assertFalse(env.expired)
        env.last_command = time.monotonic() - 1
        measured = env.data.qpos.copy()
        env.step()
        self.assertTrue(env.expired)
        np.testing.assert_array_equal(env.q_des, measured)

    def test_stale_geometry_holds_and_stale_state_sends_nothing(self):
        from run_env import Environment
        from run_controller import Controller
        env, ctrl = Environment("memq://"), Controller("memq://")
        # Connect both through one in-memory LCM instance for this offline test.
        ctrl.bus = env.bus
        env.bus.subscribe(STATE, ctrl.on_state)
        for channel in (ROBOT, STATIC):
            env.bus.subscribe(channel, ctrl.on_geometry)
        env.publish(geometry=True)
        drain(env.bus)
        ctrl.target += [.05, 0, 0]
        ctrl.update()
        drain(env.bus)
        self.assertFalse(env.expired)
        self.assertGreater(np.linalg.norm(env.dq_des), 0)
        ctrl.last_geometry[STATIC] -= 1
        ctrl.update()
        drain(env.bus)
        np.testing.assert_array_equal(env.dq_des, np.zeros(7))
        np.testing.assert_allclose(env.q_des, HOME, atol=1e-6)
        last = env.last_command
        ctrl.last_state -= 1
        ctrl.update()
        drain(env.bus)
        self.assertEqual(last, env.last_command)


if __name__ == "__main__":
    unittest.main()
