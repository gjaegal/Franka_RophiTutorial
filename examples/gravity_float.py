import numpy as np
from pylibfranka import Robot, Torques

ROBOT_IP = "192.168.0.113"

robot = Robot(ROBOT_IP)

# Start torque control
control = robot.start_torque_control()

try:
    while True:
        # Receive current robot state
        state, period = control.readOnce()
        print("q     =", state.q)
        print("dq    =", state.dq)
        print("tau_J =", state.tau_J)

        # User commanded torque = 0
        tau_d = np.zeros(7)

        command = Torques(tau_d.tolist())
        command.motion_finished = False

        control.writeOnce(command)

except KeyboardInterrupt:
    robot.stop()