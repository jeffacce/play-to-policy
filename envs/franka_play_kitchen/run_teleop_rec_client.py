# recording start/stop: grip button
# pause/resume: menu button
# calibration/freeze position: touchpad
# gripper grasp/open: trigger

import time
from .control.teleop import Teleop
from .streams.recorder import RecordingClient
from . import utils


if __name__ == "__main__":
    teleop = Teleop(utils.ROBOT_IP_ADDR)
    recording_client = RecordingClient(utils.RECORDER_IP_ADDR)

    teleop.calibrate_ref_frame()
    teleop.calibrate_start_pose()
    teleop.reset()

    last_pressed = -1
    while True:
        if teleop.buttons["touchpad"] and teleop.buttons["menu"]:
            break
        teleop.step()
        if teleop.buttons["grip"]:
            now = time.time()
            if now - last_pressed > 1.0:  # debounce
                last_pressed = now
                recording_client.cycle_state()

    recording_client.quit()
    teleop.go_home()
    teleop.close()
