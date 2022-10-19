from .streams.recorder import RecordingServer
from pathlib import Path
import logging

ROBOT_IP = "192.168.100.2"
SAVEDIR = Path("trajectories")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    server = RecordingServer(robot_ip=ROBOT_IP, directory=SAVEDIR, port=5555)
    server.run()
