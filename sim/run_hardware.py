"""pylibfranka adapter: read-only by default; --control moves to the start pose,
then enables keyboard IK motion.

The parent owns the 1 kHz robot interface. A child handles LCM and geometry at
100/10 Hz, keeping message encoding and MuJoCo out of the active control loop.
"""
import argparse
import multiprocessing as mp
from queue import Empty, Full
import signal
import struct
import time

import lcm
import numpy as np

from common import (HARDWARE_URL, HOME, MARGIN, SPEED, STATE, COMMAND, ROBOT, STATIC,
                    load_model, collision_geometry, encode_geometry, drain, IK,
                    joint_meas_t, joint_ctrl_t)

TIMEOUT = .15
MAX_SPEED = .1
MAX_ACCEL = .4
MAX_JERK = 4.
# Same pose and 20% speed as the hrg-qp hardware stack (joint 7 = 0 for its Robotiq mount).
START = np.array([0., 0., 0., -np.pi / 2, 0., np.pi / 2, 0.])
START_SPEED = .4
START_BRAKE = .75


def put_latest(queue, value):
    """Bounded best-effort transfer; a busy consumer never blocks the robot loop."""
    try:
        queue.put_nowait(value)
    except Full:
        pass


def take_latest(queue):
    value = None
    for _ in range(8):
        try:
            value = queue.get_nowait()
        except Empty:
            break
    return value


def communication(url, states, commands, done):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    model, _ = load_model()
    robot, walls = collision_geometry(model)
    bus = lcm.LCM(url)
    latest_state = None

    def receive(channel, payload):
        try:
            msg = joint_ctrl_t.decode(payload)
            # Final validation is performed against current robot state in parent.
            put_latest(commands, (time.monotonic(), msg))
        except (ValueError, TypeError, struct.error, OverflowError):
            pass

    bus.subscribe(COMMAND, receive).set_queue_capacity(1)
    last_geometry = -np.inf
    while not done.is_set():
        tick = time.monotonic()
        drain(bus)
        update = take_latest(states)
        if update is not None:
            latest_state = update
            stamp, q, dq, tau = update
            msg = joint_meas_t()
            msg.timestamp, msg.num_joints = stamp, 7
            msg.q, msg.qd, msg.tau = q, dq, tau
            bus.publish(STATE, msg.encode())
        # Do not republish an old state as though it were still arriving.
        if latest_state is not None and tick - last_geometry >= .1:
            bus.publish(ROBOT, encode_geometry(robot, latest_state[0]))
            bus.publish(STATIC, encode_geometry(walls, latest_state[0]))
            last_geometry = tick
        done.wait(max(0, .005 - (time.monotonic() - tick)))


class VelocityGuard:
    """Validate references, expire old packets, and shape bounded joint velocities."""
    def __init__(self, limits):
        self.limits = limits
        self.desired = np.zeros(7)
        self.last_receive = -np.inf
        self.last_stamp = -np.inf

    def accept(self, received, msg, state, stamp, now):
        q, dq = np.array(msg.q_des), np.array(msg.qd_des)
        if (not msg.valid or msg.num_joints != 7 or q.shape != (7,) or dq.shape != (7,)
                or not np.isfinite(np.r_[q, dq, msg.timestamp]).all()
                or not 0 <= now - received <= TIMEOUT
                or not -.005 <= stamp - msg.timestamp <= TIMEOUT
                or msg.timestamp <= self.last_stamp
                or np.max(np.abs(dq)) > SPEED + .001
                or np.max(np.abs(q - np.array(state.q))) > .05
                or np.any(q < self.limits[:, 0] + .02)
                or np.any(q > self.limits[:, 1] - .02)):
            return False
        # Uniform scaling preserves the QP's joint direction.
        self.desired = dq * min(1., MAX_SPEED / max(np.max(np.abs(dq)), 1e-12))
        self.last_receive, self.last_stamp = received, msg.timestamp
        return True

    def velocity(self, state, dt, now, stopping=False):
        if not np.isfinite(dt) or not 0 < dt <= .005:
            raise RuntimeError(f"Robot cycle duration out of range: {dt}")
        q, v, a = map(np.asarray, (state.q, state.dq_d, state.ddq_d))
        if not np.isfinite(np.r_[q, v, a]).all():
            raise RuntimeError("Non-finite robot state")
        if np.max(np.abs(v)) > MAX_SPEED + .001 or np.max(np.abs(a)) > MAX_ACCEL + .01:
            raise RuntimeError("Robot desired rate exceeds the adapter's configured limits")
        if np.any(q < self.limits[:, 0] + .02) or np.any(q > self.limits[:, 1] - .02):
            raise RuntimeError("Robot reached the configured joint-range margin")
        target = self.desired if now - self.last_receive <= TIMEOUT and not stopping else np.zeros(7)
        # Begin braking before the hard margin, including acceleration/jerk lag.
        if np.any(q + v * .5 < self.limits[:, 0] + .04) or np.any(q + v * .5 > self.limits[:, 1] - .04):
            target = np.zeros(7)
        # Conservative form of libfranka's velocity-rate limiting equations:
        # bound jerk, then bound acceleration with room to decelerate at speed limits.
        # Proportional velocity error avoids bang-bang oscillation near zero.
        wanted_acceleration = np.clip(8. * (target - v), -MAX_ACCEL, MAX_ACCEL)
        acceleration = a + np.clip(wanted_acceleration - a, -MAX_JERK * dt, MAX_JERK * dt)
        upper = np.minimum(MAX_ACCEL, (MAX_JERK / MAX_ACCEL) * (MAX_SPEED - v))
        lower = np.maximum(-MAX_ACCEL, (MAX_JERK / MAX_ACCEL) * (-MAX_SPEED - v))
        acceleration = np.clip(acceleration, lower, upper)
        return v + acceleration * dt


def pose_is_clear(ik, q):
    ik.set_state(q, np.zeros(7))
    return min(d for d, _ in ik.collision_rows()) >= MARGIN


class StartMove:
    """Quintic joint path to START. A stop request eases to rest along that path."""
    def __init__(self, q):
        self.q, self.delta = np.array(q), START - q
        # A quintic peaks at 1.875 d/T; the floor keeps short moves gentle.
        self.duration = max(1.5, 1.875 * np.max(np.abs(self.delta)) / START_SPEED)
        self.elapsed = self.braking = 0.

    def step(self, dt, stopping):
        if stopping:
            self.braking = min(1., self.braking + dt / START_BRAKE)
        # Slowing time rather than the joints keeps a stop on the planned path.
        self.elapsed += dt * (1 - self.braking ** 2 * (3 - 2 * self.braking))
        x = min(1., self.elapsed / self.duration)
        q = self.q + self.delta * x ** 3 * (10 + x * (6 * x - 15))
        return q, x == 1 or self.braking == 1


def move_to_start(active, positions, stop_requested):
    move = None
    while True:
        state, period = active.readOnce()
        # libfranka requires a position motion to begin at the commanded pose.
        move = move or StartMove(state.q_d)
        q, finished = move.step(period.to_sec(), stop_requested())
        command = positions(q.tolist())
        command.motion_finished = finished
        active.writeOnce(command)
        if finished:
            return


class MockRobot:
    """Paced kinematic backend for end-to-end tests. Never opens a robot socket."""
    def __init__(self):
        from types import SimpleNamespace
        self.state = SimpleNamespace(q=HOME.tolist(), q_d=HOME.tolist(), dq=[0.] * 7, tau_J=[0.] * 7,
                                     dq_d=[0.] * 7, ddq_d=[0.] * 7)
        self.next_tick = time.monotonic()
        self.active = False

    def read_once(self):
        time.sleep(.001)
        return self.state

    def start_joint_position_control(self, mode):
        return self.start_joint_velocity_control(mode)

    def start_joint_velocity_control(self, mode):
        if self.active:
            raise RuntimeError("Mock robot is already under control")
        self.active = True
        self.next_tick = time.monotonic()
        return self

    def readOnce(self):
        from types import SimpleNamespace
        self.next_tick += .001
        time.sleep(max(0, self.next_tick - time.monotonic()))
        return self.state, SimpleNamespace(to_sec=lambda: .001)

    def writeOnce(self, command):
        v = ((np.array(command.q) - self.state.q) / .001 if hasattr(command, "q")
             else np.array(command.dq))
        old = np.array(self.state.dq_d)
        self.state.q = self.state.q_d = (np.array(self.state.q) + v * .001).tolist()
        self.state.dq = self.state.dq_d = v.tolist()
        self.state.ddq_d = ((v - old) / .001).tolist()
        self.active = not command.motion_finished

    def stop(self):
        self.active = False


def run(args):
    import pylibfranka as franka
    model, _ = load_model()
    if args.mock:
        robot = MockRobot()
    elif args.control:
        # Motion control must retain libfranka's real-time scheduling check.
        robot = franka.Robot(args.ip)
    else:
        # State inspection does not run a user control loop.
        robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
    # Read-only mode never starts a motion generator.
    initial = robot.read_once()
    if (len(initial.q) != 7 or len(initial.dq) != 7
            or not np.isfinite([*initial.q, *initial.dq]).all()):
        raise RuntimeError("Invalid initial robot state")
    if args.control:
        ik = IK(model)
        ik.robot, ik.walls = collision_geometry(model)
        limits = model.jnt_range
        if (np.any(np.array(initial.q) < limits[:, 0] + .05)
                or np.any(np.array(initial.q) > limits[:, 1] - .05)
                or np.max(np.abs(initial.dq)) > .01
                or np.max(np.abs(initial.dq_d)) > 1e-5
                or np.max(np.abs(initial.ddq_d)) > .01):
            raise RuntimeError("Start control at rest and away from joint limits")
    context = mp.get_context("spawn")
    states, commands = context.Queue(maxsize=2), context.Queue(maxsize=2)
    done = context.Event()
    worker = context.Process(target=communication, args=(args.url, states, commands, done))
    worker.start()
    requested_stop = False

    def request_stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    active = None
    completed = False
    start, stamp, last_publish = time.monotonic(), 0., -np.inf
    guard = VelocityGuard(model.jnt_range.copy())
    try:
        print(f"{'MOCK' if args.mock else args.ip}: {'VELOCITY CONTROL' if args.control else 'READ ONLY'}; URL={args.url}", flush=True)
        if args.control:
            if args.move_to_start:
                print(f"Moving to the start pose {np.round(START, 3).tolist()} ...", flush=True)
                active = robot.start_joint_position_control(franka.ControllerMode.JointImpedance)
                # State stays unpublished so a running controller cannot take a mid-path target.
                move_to_start(active, franka.JointPositions, lambda: requested_stop)
                active = None
                if requested_stop:
                    return
                if np.max(np.abs(np.array(robot.read_once().q) - START)) > .02:
                    print("Did not reach the start pose; control starts where the arm is.", flush=True)
                start = time.monotonic()
            # The start move may begin outside the cell; velocity control may not.
            if not pose_is_clear(ik, robot.read_once().q):
                raise RuntimeError("Pose violates the virtual cell clearance; reposition before control")
            active = robot.start_joint_velocity_control(franka.ControllerMode.JointImpedance)
        while True:
            if active is None:
                if requested_stop or time.monotonic() - start >= args.seconds:
                    break
                state = robot.read_once()
                stamp = time.monotonic() - start
            else:
                state, period = active.readOnce()
                dt = period.to_sec()
                # libfranka's first period is zero: first command must remain at rest.
                stamp += dt
            now = time.monotonic()
            if now - last_publish >= .01:
                put_latest(states, (stamp, list(state.q), list(state.dq), list(state.tau_J)))
                last_publish = now
            if not worker.is_alive():
                raise RuntimeError("LCM worker exited")
            if active is None:
                time.sleep(.005)
                continue
            pending = take_latest(commands)
            if pending is not None:
                guard.accept(*pending, state, stamp, now)
            stopping = requested_stop or now - start >= args.seconds
            velocity = np.zeros(7) if dt == 0 and stamp == 0 else guard.velocity(state, dt, now, stopping)
            command = franka.JointVelocities(velocity.tolist())
            finished = stopping and np.max(np.abs(velocity)) < 1e-5 and np.max(np.abs(state.ddq_d)) < .01
            if finished:
                command.motion_finished = True
            active.writeOnce(command)
            if finished:
                completed = True
                break
    finally:
        try:
            if active is not None and not completed:
                robot.stop()
        finally:
            done.set()
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            for queue in (states, commands):
                queue.cancel_join_thread()
                queue.close()
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.0.113")
    parser.add_argument("--url", default=HARDWARE_URL)
    parser.add_argument("--control", action="store_true", help="Enable actual joint-velocity motion")
    parser.add_argument("--move-to-start", action=argparse.BooleanOptionalAction, default=True,
                        help="With --control, first move to the start pose")
    parser.add_argument("--mock", action="store_true", help="Use a kinematic test robot instead of hardware")
    parser.add_argument("--seconds", type=float, default=float("inf"))
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        parser.exit(1, f"Hardware adapter stopped: {exc}\n")


if __name__ == "__main__":
    main()
