"""Keyboard target + collision-constrained velocity IK, at 100 Hz."""
import argparse
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager

import lcm
import mujoco
import numpy as np

from common import (HOME, URL, HARDWARE_URL, DT, STATE, COMMAND, TARGET, ROBOT, STATIC,
                    load_model, decode_geometry, drain, command, IK,
                    joint_meas_t, pose3d_t)


@contextmanager
def keyboard(enabled):
    old = termios.tcgetattr(sys.stdin) if enabled else None
    try:
        if enabled:
            tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        if old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


class Controller:
    def __init__(self, url, hardware=False):
        model, _ = load_model()
        self.ik = IK(model, speed=.1, cart_speed=.04) if hardware else IK(model)
        self.bus = lcm.LCM(url)
        self.target = None
        self.q_des = None
        self.stamp = 0.
        self.last_state = -np.inf
        self.last_geometry = {ROBOT: -np.inf, STATIC: -np.inf}
        self.paused = False
        self.step_size = .01
        self.bus.subscribe(STATE, self.on_state).set_queue_capacity(1)
        for channel in (ROBOT, STATIC):
            self.bus.subscribe(channel, self.on_geometry).set_queue_capacity(1)

    def on_state(self, channel, payload):
        msg = joint_meas_t.decode(payload)
        if msg.num_joints != 7 or not np.isfinite([*msg.q, *msg.qd, msg.timestamp]).all():
            return
        # Simulator or adapter restarts reset the clock, the integrated position
        # reference and the target: the arm may no longer be where it was.
        if msg.timestamp < self.stamp:
            self.q_des = None
            self.target = None
        self.ik.set_state(msg.q, msg.qd)
        self.stamp = msg.timestamp
        self.last_state = time.monotonic()
        if self.target is None:
            self.target = self.ik.tip()[0]
        if self.q_des is None:
            self.q_des = np.array(msg.q)

    def on_geometry(self, channel, payload):
        items = decode_geometry(payload)
        if channel == ROBOT:
            self.ik.robot = items
        else:
            self.ik.walls = items
        self.last_geometry[channel] = time.monotonic()

    def key(self, key):
        if key == "p":
            self.paused = not self.paused
        if key == "[":
            self.step_size = max(.002, self.step_size / 2)
        if key == "]":
            self.step_size = min(.05, self.step_size * 2)
        if self.target is not None:
            directions = {"w": (0, 1), "s": (0, -1), "a": (1, 1),
                          "d": (1, -1), "r": (2, 1), "f": (2, -1)}
            if key in directions:
                axis, sign = directions[key]
                self.target[axis] += sign * self.step_size
            if key == "h":
                # Cartesian home target; it still goes through the collision QP.
                home = mujoco.MjData(self.ik.model)
                home.qpos[:] = HOME
                mujoco.mj_forward(self.ik.model, home)
                self.target = home.site("index_tip").xpos.copy()

    def update(self):
        now = time.monotonic()
        if now - self.last_state > .15:
            self.q_des = None
            return
        ready = all(now - t < .5 for t in self.last_geometry.values())
        if self.paused or not ready:
            dq = np.zeros(7)
            self.q_des = self.ik.data.qpos.copy()
        else:
            dq = self.ik.solve(self.target)
            # A stopped/failed plan must not leave an old position error to chase.
            if not np.any(dq):
                self.q_des = self.ik.data.qpos.copy()
            else:
                self.q_des += DT * dq
                # Bound servo tracking error to avoid windup against contacts.
                self.q_des = np.clip(self.q_des, self.ik.data.qpos - .03, self.ik.data.qpos + .03)
        self.q_des = np.clip(self.q_des, self.ik.model.jnt_range[:, 0] + .02,
                             self.ik.model.jnt_range[:, 1] - .02)
        self.bus.publish(COMMAND, command(self.q_des, dq, self.stamp))
        msg = pose3d_t()
        msg.timestamp = self.stamp
        msg.position = self.target.tolist()
        tip, J = self.ik.tip()
        msg.tip, msg.velocity = tip.tolist(), (J @ dq).tolist()
        self.bus.publish(TARGET, msg.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true",
                        help="Use the hardware URL and lower IK speed limits")
    parser.add_argument("--url", help="Override the simulation/hardware LCM URL")
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--seconds", type=float, default=float("inf"))
    parser.add_argument("--offset", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Initial target offset in metres, useful for headless checks")
    args = parser.parse_args()
    enabled = not args.no_keyboard
    if enabled and not sys.stdin.isatty():
        parser.error("Use an interactive terminal or --no-keyboard")
    ctrl = Controller(args.url or (HARDWARE_URL if args.hardware else URL), args.hardware)
    print("w/s: +/-x   a/d: +/-y   r/f: +/-z   [/]: step   h: home target   p: pause   q: quit", flush=True)
    start, report = time.monotonic(), 0.
    offset_applied = False
    with keyboard(enabled):
        try:
            while time.monotonic() - start < args.seconds:
                tick = time.monotonic()
                drain(ctrl.bus)
                if ctrl.target is not None and args.offset and not offset_applied:
                    ctrl.target += np.array(args.offset)
                    offset_applied = True
                if enabled:
                    while select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        if key == "q":
                            return
                        ctrl.key(key)
                ctrl.update()
                if tick - report >= 1:
                    if ctrl.target is None or tick - ctrl.last_state > .15:
                        print("Waiting for robot state...", flush=True)
                    else:
                        error = np.linalg.norm(ctrl.target - ctrl.ik.tip()[0])
                        print(f"error={error:.3f} m | {ctrl.ik.status} | paused={ctrl.paused}", flush=True)
                    report = tick
                time.sleep(max(0, DT - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            pass
        finally:
            if ctrl.target is not None:
                ctrl.paused = True
                ctrl.update()


if __name__ == "__main__":
    main()
