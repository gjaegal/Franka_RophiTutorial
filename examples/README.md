# Basic FR3 examples with pylibfranka

These Python examples use pylibfranka to operate a **real robot**. They have not been tested on hardware. The Robotiq gripper is mounted payload; these examples do not control it.

## Setup

From the project root, activate the Python 3.12 environment and install the Python dependencies:

```bash
source .venv/bin/activate
python -m pip install pylibfranka numpy
cd examples
```

Confirm that the installed pylibfranka version supports the robot system software and FCI setup. Configure the mounted gripper, adapter, and payload mass properties correctly. Follow [the hardware plan](../02_real_robot_plan.md) before running control examples.

## Examples

- `read_state.py`: continuously prints measured joint positions, velocities, and torques. It uses the `ROBOT_IP` address directly in the `Robot(...)` call; edit that address before running.
- `gravity_float.py`: continuously sends zero additional joint torque. Set `ROBOT_IP` before running. The arm can move and drift.
- `pd_control.py`: applies joint PD torques to hold the initial measured configuration. Set `ROBOT_IP` and review the gains before running.
- `joint_position_example.py`: commands joint position motion and accepts the robot address through `--ip`. It configures collision thresholds and prompts before starting motion; review its settings and trajectory first.

Begin with the read-only example:

```bash
python read_state.py
```

After the hardware checklist and an operator-approved test, run the desired control example:

```bash
python gravity_float.py
python pd_control.py
python joint_position_example.py --ip <robot-ip>
```

Run only one control process at a time. The torque examples run until interrupted with Ctrl+C. Franka Control supplies gravity and motor-friction compensation for the configured model; user torque is **additional** torque. Do not add the full model gravity again. Python control-loop timing must meet the installed system's requirements; these scripts have not been validated for real-time operation.
