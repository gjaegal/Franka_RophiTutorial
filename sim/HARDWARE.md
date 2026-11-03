# Real FR3 control with a simultaneous MuJoCo mirror

Use the same keyboard IK controller, with pylibfranka as the plant. The MuJoCo window follows **measured** joints from the real arm. It never integrates an independent simulated trajectory or connects to the robot.

```text
keyboard / OSQP (100 Hz) -- joint velocity reference --> hardware adapter --> FR3 (1 kHz)
       ^                                                  |
       +--------- measured joints + collision geometry ----+
                                                          |
                                    measured joints ------+--> MuJoCo mirror (30 Hz)
keyboard target ------------------------------------------------> red sphere
```

## Setup

```bash
cd /home/glen/workspace/fr3_robotiq_gravity_comp
source .venv/bin/activate
python -m pip install -r sim/requirements-hardware.txt
```

The requirements match the currently installed pylibfranka 0.21.3. Confirm compatibility with the robot's installed system software and FCI configuration. The workstation must support the robot's 1 kHz control timing; the adapter retains pylibfranka's default real-time enforcement. Robot collision thresholds, load, impedance settings, and error recovery are left to the existing robot configuration.

This controls the virtual `index_tip` already used in simulation: the FR3 attachment point 0.107 m along link 7's local z axis, **not a Robotiq fingertip**. The scene and collision primitives still contain the bare FR3. Configure any mounted gripper/payload on the robot, and extend the model before relying on gripper clearance. The virtual enclosure is fixed in the robot base frame; it is not measured from the physical room. Its dimensions are set in `common.py`.

## First inspect measured motion without commanding anything

Terminal 1:

```bash
cd /home/glen/workspace/fr3_robotiq_gravity_comp
source .venv/bin/activate
python sim/run_hardware.py --ip 192.168.0.113
```

Terminal 2:

```bash
cd /home/glen/workspace/fr3_robotiq_gravity_comp
source .venv/bin/activate
python sim/run_mirror.py
```

Replace the IP with the robot's configured address. The window should show its actual pose and `LIVE`. If state stops arriving, it freezes at the last measured pose and shows `STALE`. Before any state arrives, the displayed home pose is explicitly marked as a placeholder. This mode calls `read_once()` only; it never starts a motion generator.

Stop terminal 1 with Ctrl+C before starting the control adapter below. Keep the mirror open.

## Move the real robot with the keyboard

With the robot at rest, FCI enabled, its path checked, and the operator ready to stop it, start the adapter in terminal 1:

```bash
python sim/run_hardware.py --ip 192.168.0.113 --control
```

**The arm moves as soon as this command runs, so keep the user stop in hand.** The adapter first moves to the start pose `[0, 0, 0, -90°, 0, 90°, 0]`, the same pose and 20% speed (up to 0.4 rad/s per joint) as the hrg-qp hardware stack, and only then starts velocity control. The move follows a straight joint path with no virtual enclosure or obstacle checks, so check that path yourself. Enclosure clearance is enforced once it finishes, before velocity control starts. Ctrl+C during the move eases to rest along the path and exits. Pass `--no-move-to-start` to begin where the arm already is.

Leave terminal 2 running `run_mirror.py`. Start terminal 3:

```bash
cd /home/glen/workspace/fr3_robotiq_gravity_comp
source .venv/bin/activate
python sim/run_controller.py --hardware
```

Focus terminal 3. The red target starts at the measured tool point, so starting does not command a move to the simulation home pose. `w/s`, `a/d`, and `r/f` nudge the target along base-frame x/y/z. `p` pauses, `q` quits the controller, and Ctrl+C in the hardware terminal ends velocity control after decelerating. `h` sets the target to the simulation's Cartesian home point and therefore **can command a substantial real move**.

Run only one robot-owning process. Stop `read_state.py`, other example scripts, and any other FCI controller before starting this adapter. Use `run_mirror.py` for the hardware display; `run_env.py` remains a separate simulation plant on its own URL.

## How each file works

| File | Responsibility |
| --- | --- |
| `run_hardware.py` | Owns the pylibfranka connection. Read-only unless `--control` is present. In control mode moves to the start pose with `start_joint_position_control`, publishing nothing until it arrives, then starts `start_joint_velocity_control(ControllerMode.JointImpedance)`, receives state with `readOnce()`, and writes `JointVelocities`. A child process handles LCM and geometry so serialization and MuJoCo model processing stay out of the active loop. |
| `run_controller.py --hardware` | Reuses the simulation QP with reduced limits: 0.04 m/s Cartesian target tracking and 0.1 rad/s per joint. Uses the latest measured FR3 joints for Jacobians and collision constraints. |
| `run_mirror.py` | Subscribes to joint state and target. Assigns `qpos`/`qvel` from measurements and calls `mj_forward()` for drawing. Draws the arm translucent so the red target stays visible at the flange. Never calls `mj_step()` or publishes joint commands. |
| `common.py` | Shared model, Jacobians, geometry and LCM message formats. Defines simulation and hardware transport URLs. |
| `test_hardware.py` | Offline validation of packet rejection, smoothing bounds, stopping, stale commands, joint margins and measured-joint visualization. |
| `requirements-hardware.txt` | Simulation dependencies plus pylibfranka. |

The same five channel names are used as in simulation, on hardware URL `udpm://239.255.76.67:7669?ttl=0` (simulation uses port 7668). All three hardware scripts accept `--url` if an override is needed. The controller selects the hardware default with `--hardware`; keep that flag even with a custom URL to retain its reduced limits.

## Command and stop behavior

- The adapter consumes `qd_des` from `joint_ctrl_t`. `q_des` is checked for reasonable proximity and joint limits, but is not sent to the robot as a position setpoint. Position integration in the controller is retained for simulation compatibility.
- Accepted packets must be finite, valid, seven-joint commands, recent in both receive time and echoed robot-state time, newer than the previous accepted packet, and within the configured ranges. Old, replayed, oversized, or malformed references are ignored.
- Output joint speed is limited to 0.1 rad/s, acceleration to 0.4 rad/s² and jerk to 4 rad/s³ for a normal, stationary start. The adapter uses measured desired velocity/acceleration (`dq_d`, `ddq_d`) to produce continuous velocity references. It does not assume `writeOnce()` applies an automatic rate limiter.
- On controller timeout (150 ms), output ramps toward zero and remains in active joint impedance control. Normal Ctrl+C/termination also ramps to zero, then sets `motion_finished`. An adapter error, communication-process failure, invalid robot cycle or joint-margin violation calls `robot.stop()` and exits. No automatic error recovery or restart is attempted.
- Startup rejects moving poses and poses near the model's joint limits. Just before velocity control starts, after any start move, it rejects a measured pose that violates the virtual enclosure margin. With `--control` it drives the robot to the start pose unless `--no-move-to-start` is given; read-only mode never moves it. A controller left running across an adapter restart drops its old target.

These limits are application limits, not replacements for the robot's position-dependent limits or safety functions. Smoothing, sensing delay and hardware tracking mean the executed velocity differs from the QP result. The current QP checks arm-versus-cell collisions with conservative spheres; it does not constrain self-collisions, unmodeled tools, or real obstacles absent from the cell model. Python timing and the actual hardware stop behavior still require validation on your workstation and robot.

The implementation follows the installed bindings and the version-matched [pylibfranka velocity example](https://github.com/frankarobotics/libfranka/blob/0.21.3/pylibfranka/examples/joint_velocity_example.py), [active motion implementation](https://github.com/frankarobotics/libfranka/blob/0.21.3/src/active_motion_generator.cpp), and [rate-limit equations](https://github.com/frankarobotics/libfranka/blob/0.21.3/src/rate_limiting.cpp). Franka supplies the joint impedance controller and gravity compensation; this adapter sends no additional gravity torque.

## Test the whole path without a real robot

Use the same three terminals, replacing the hardware adapter command with:

```bash
python sim/run_hardware.py --mock --control
```

Keep `python sim/run_mirror.py` and `python sim/run_controller.py --hardware` unchanged. `--mock` selects a paced kinematic joint integrator and never opens a robot connection. It checks communication, keyboard targeting, velocity shaping and the mirror, rather than robot dynamics.

Offline checks:

```bash
python sim/test_sim.py
python sim/test_hardware.py
```

Verified without hardware: eight simulation tests and fifteen hardware-boundary tests pass. A three-process mock run received all five channels, tracked a 5 cm target offset to below 1 mm error, and stopped after controller exit. No real-robot connection or motion has been performed during implementation.
