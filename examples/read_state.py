import time
import numpy as np
from pylibfranka import RealtimeConfig, Robot

robot = Robot("192.168.0.113", RealtimeConfig.kIgnore)

while True:
    state = robot.read_once()
    print("q     :", np.round(state.q, 3))
    print("dq    :", np.round(state.dq, 3))
    print("tau_J :", np.round(state.tau_J, 3))
    print()
    time.sleep(0.1)