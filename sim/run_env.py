"""MuJoCo plant + viewer. No physical robot connection is made."""
import argparse
import time
from contextlib import nullcontext

import lcm
import mujoco
import numpy as np

from common import (HOME, URL, SPEED, STATE, COMMAND, TARGET, ROBOT, STATIC,
                    load_model, collision_geometry, encode_geometry, drain,
                    joint_meas_t, joint_ctrl_t, pose3d_t)


class Environment:
    def __init__(self, url):
        self.model, self.data = load_model()
        self.bus = lcm.LCM(url)
        self.q_des = HOME.copy()
        self.dq_des = np.zeros(7)
        self.last_command = -np.inf
        self.last_stamp = -np.inf
        self.robot, self.walls = collision_geometry(self.model)
        self.bus.subscribe(COMMAND, self.on_command).set_queue_capacity(1)
        self.bus.subscribe(TARGET, self.on_target).set_queue_capacity(1)
        self.expired = True

    def on_command(self, channel, payload):
        msg = joint_ctrl_t.decode(payload)
        q, dq = np.array(msg.q_des), np.array(msg.qd_des)
        if (not msg.valid or msg.num_joints != 7 or q.shape != (7,) or dq.shape != (7,)
                or not np.isfinite(np.r_[q, dq, msg.timestamp]).all()
                or abs(msg.timestamp - self.data.time) > .15
                or msg.timestamp < self.last_stamp
                or np.max(np.abs(dq)) > SPEED + .001):
            return
        limits = self.model.jnt_range
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            return
        self.q_des, self.dq_des = q, dq
        self.last_stamp = msg.timestamp
        self.last_command = time.monotonic()
        self.expired = False

    def on_target(self, channel, payload):
        msg = pose3d_t.decode(payload)
        if np.isfinite(msg.position).all():
            self.data.mocap_pos[0] = msg.position

    def step(self):
        # If the controller disappears, hold once at the current measured pose.
        if time.monotonic() - self.last_command > .15 and not self.expired:
            self.q_des = self.data.qpos.copy()
            self.dq_des[:] = 0
            self.expired = True
        # Native position servos plus bias compensation; not a model of FCI internals.
        self.data.ctrl[:] = self.q_des
        self.data.qfrc_applied[:] = self.data.qfrc_bias
        mujoco.mj_step(self.model, self.data)

    def publish(self, geometry=False):
        msg = joint_meas_t()
        msg.timestamp, msg.num_joints = self.data.time, 7
        msg.q = self.data.qpos.tolist()
        msg.qd = self.data.qvel.tolist()
        msg.tau = self.data.qfrc_actuator.tolist()
        self.bus.publish(STATE, msg.encode())
        if geometry:
            self.bus.publish(ROBOT, encode_geometry(self.robot, self.data.time))
            self.bus.publish(STATIC, encode_geometry(self.walls, self.data.time))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=float("inf"))
    args = parser.parse_args()
    env = Environment(args.url)
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(env.model, env.data)
    # print("Simulation only. Use the controller terminal for keyboard input.", flush=True)
    with context as viewer:
        if viewer:
            viewer.cam.lookat[:] = [0, 0, .65]
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
        tick = 0
        start = time.monotonic()
        try:
            while env.data.time < args.seconds and (viewer is None or viewer.is_running()):
                drain(env.bus)
                env.step()
                if tick % 5 == 0:
                    env.publish(geometry=tick % 50 == 0)
                if viewer and tick % 16 == 0:
                    viewer.sync()
                tick += 1
                time.sleep(max(0, start + tick * env.model.opt.timestep - time.monotonic()))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
