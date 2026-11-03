import numpy as np
from pylibfranka import Robot, Torques

ROBOT_IP = "172.16.0.2"

robot = Robot(ROBOT_IP)

# --------------------------------
# Initial state
# --------------------------------
state = robot.read_once()

q_initial = np.array(state.q)

# Hold the position where the controller starts
q_des = q_initial.copy()

# --------------------------------
# PD gains
# --------------------------------
Kp = np.array([
    50.0, 50.0, 50.0, 50.0,
    30.0, 20.0, 10.0
])

Kd = np.array([
    10.0, 10.0, 10.0, 10.0,
    6.0, 5.0, 3.0
])

# --------------------------------
# Start torque control
# --------------------------------
control = robot.start_torque_control()

try:
    while True:

        # Read state
        state, period = control.readOnce()

        q  = np.array(state.q)
        dq = np.array(state.dq)

        # --------------------------------
        # PD controller
        #
        # tau = Kp(q_des - q) - Kd*dq
        # --------------------------------
        tau_pd = Kp * (q_des - q) - Kd * dq

        # Send ONLY PD torque
        # Gravity + friction are compensated internally
        command = Torques(tau_pd.tolist())
        command.motion_finished = False

        control.writeOnce(command)

except KeyboardInterrupt:
    robot.stop()