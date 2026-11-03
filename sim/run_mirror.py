"""Display measured FR3 joints and target. This process never sends robot commands."""
import argparse
import time

import lcm
import mujoco
import numpy as np

from common import HARDWARE_URL, STATE, TARGET, load_model, drain, joint_meas_t, pose3d_t


class Mirror:
    def __init__(self, url):
        self.model, self.data = load_model()
        # The target sits on the flange face, so a solid arm hides it from most views.
        self.model.mat_rgba[:, 3] = .2
        self.bus = lcm.LCM(url)
        self.last_state = -np.inf
        self.have_state = False
        self.have_target = False
        self.bus.subscribe(STATE, self.on_state).set_queue_capacity(1)
        self.bus.subscribe(TARGET, self.on_target).set_queue_capacity(1)

    def on_state(self, channel, payload):
        msg = joint_meas_t.decode(payload)
        if (msg.num_joints != 7 or len(msg.q) != 7 or len(msg.qd) != 7
                or not np.isfinite([*msg.q, *msg.qd]).all()):
            return
        self.data.qpos[:] = msg.q
        self.data.qvel[:] = msg.qd
        mujoco.mj_forward(self.model, self.data)
        if not self.have_target:
            self.data.mocap_pos[0] = self.data.site("index_tip").xpos
        self.last_state = time.monotonic()
        self.have_state = True

    def on_target(self, channel, payload):
        msg = pose3d_t.decode(payload)
        if np.isfinite(msg.position).all():
            self.data.mocap_pos[0] = msg.position
            self.have_target = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=HARDWARE_URL)
    parser.add_argument("--seconds", type=float, default=float("inf"))
    parser.add_argument("--headless", action="store_true", help="Test subscriptions without a window")
    args = parser.parse_args()
    mirror = Mirror(args.url)
    from contextlib import nullcontext
    if args.headless:
        context = nullcontext(None)
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(mirror.model, mirror.data)
    print("Measured-joint mirror only; no physics stepping or robot commands.", flush=True)
    start, last_report = time.monotonic(), -np.inf
    previous_status = None
    with context as viewer:
        if viewer:
            viewer.cam.lookat[:] = [0, 0, .65]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 135, -25
        try:
            while time.monotonic() - start < args.seconds and (viewer is None or viewer.is_running()):
                tick = time.monotonic()
                drain(mirror.bus)
                status = ("LIVE" if tick - mirror.last_state < .25 else
                          "STALE — displaying last measured pose" if mirror.have_state else
                          "WAITING — displayed pose is a placeholder")
                if status != previous_status and tick - last_report > .25:
                    print(status, flush=True)
                    if viewer:
                        viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Robot state", status)])
                    previous_status, last_report = status, tick
                if viewer:
                    viewer.sync()
                time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
