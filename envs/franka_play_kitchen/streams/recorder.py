import zmq
import time
import uuid
import pickle
import logging
import threading
from . import core
from enum import Enum
from pathlib import Path
import multiprocessing as mp
from typing import Optional, Any


class RecorderWorker(mp.Process):
    def __init__(
        self,
        stop_event: mp.Event,
        stream: core.AbstractStream,
        filepath: Path,
        freq_limit: Optional[float] = None,
    ):
        super().__init__()
        self.buffer = []
        self.stream = stream
        self.filepath = filepath
        self.last_read = -1
        self.period = -1
        self.exit = stop_event
        if freq_limit is not None:
            self.period = 1.0 / freq_limit

    def run(self):
        if self.period == -1:  # no frequency limit
            while not self.exit.is_set():
                self.buffer.append(self.stream.get_data())
        else:
            while not self.exit.is_set():
                now = time.time()
                if now - self.last_read > self.period:
                    try:
                        data = self.stream.get_data()
                        self.last_read = now
                    except:
                        data = None
                    if data is not None:
                        self.buffer.append(data)
        self.stream.close()
        self.stream.save_buffer(self.buffer, self.filepath)


class Recorder:
    def __init__(
        self,
        stream: core.AbstractStream,
        filepath: Path,
        freq_limit: Optional[float] = None,
    ):
        self.buffer = []
        self.stream = stream
        self.filepath = filepath
        self.freq_limit = freq_limit
        self.stop_event = mp.Event()
        self.p = RecorderWorker(
            self.stop_event, stream, filepath, freq_limit=freq_limit
        )

    def start(self):
        self.p.start()

    def stop(self):
        self.stop_event.set()

    def join(self):
        self.p.join()

    def __str__(self):
        return f"Recorder(stream={self.stream}, filepath={self.filepath}, freq_limit={self.freq_limit}, len(buffer): {len(self.buffer)})"


class IPCRecorderWorker(mp.Process):
    def __init__(
        self,
        socket_path: str,
        stop_event: mp.Event,
        stream: core.AbstractStream,
        freq_limit: Optional[float] = None,
    ):
        super().__init__()
        self.socket_path = socket_path
        self.stream = stream
        self.last_read = -1
        self.period = -1
        self.exit = stop_event
        if freq_limit is not None:
            self.period = 1.0 / freq_limit

    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("ipc://%s" % self.socket_path)

        if self.period == -1:  # no frequency limit
            while not self.exit.is_set():
                self._send_data(self.stream.get_data())
        else:
            while not self.exit.is_set():
                now = time.time()
                if now - self.last_read > self.period:
                    self.last_read = now
                    self._send_data(self.stream.get_data())
        self.stream.close()
        self.socket.close()
        self.context.term()

    def _send_data(self, data: Any):
        self.socket.send(b"data " + pickle.dumps(data, protocol=-1))


class IPCRecorder(threading.Thread):
    def __init__(
        self,
        stream: core.AbstractStream,
        freq_limit: Optional[float] = None,
        manual_get_data: bool = False,  # for only pubsub streaming w/o recording
    ):
        super().__init__()
        self.buffer = []
        self.exit = False
        self.stream = stream
        self.freq_limit = freq_limit
        self.manual_get_data = manual_get_data
        self.stop_event = mp.Event()
        self.socket_path = "/tmp/%s" % uuid.uuid4()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)  # drop old messages
        self.socket.connect("ipc://%s" % self.socket_path)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"data")

        self.p = IPCRecorderWorker(
            self.socket_path, self.stop_event, stream, freq_limit=freq_limit
        )

    def run(self):
        self.p.start()
        if not self.manual_get_data:
            while not self.exit:
                data = self.get_data()
                if data is not None:
                    self.buffer.append(data)

    def stop(self):
        self.stop_event.set()
        self.exit = True
        self.socket.close()
        self.context.term()
        self.p.join()

    def get_data(self):
        try:
            msg = self.socket.recv()
            data = msg.lstrip(b"data ")
            data = pickle.loads(data)
        except zmq.error.ZMQError:
            data = None
        return data


class RecordingServerState(Enum):
    IDLE = 0
    ROLLING = 1
    RECORDING = 2
    HALTED = 3


class RecordingServer:
    def __init__(self, robot_ip: str, directory: Path, port=5555):
        self.context = zmq.Context()
        self.socket: zmq.Socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.start_time = None
        self.stop_time = None
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state = RecordingServerState.IDLE
        self.robot_ip = robot_ip
        self.all_recs = []
        logging.info(f"Recording server started on port {port}")

    def reply(self, msg: str, log=True):
        if log:
            logging.info(f">> {msg}")
        self.socket.send_string(msg)

    def run(self):
        while True:
            message = self.socket.recv_string()
            logging.info(f"<< {message}")
            if "PING" in message:
                self.reply("Server is running")
            elif "ROLL" in message:
                if self.state == RecordingServerState.IDLE:
                    self.roll()
                else:
                    self.reply("Invalid; state={}".format(self.state))
            elif "START" in message:
                # message: "START:<timestamp>"
                if self.state == RecordingServerState.ROLLING:
                    timestamp = float(message.split(":")[1])
                    self.start(timestamp)
                else:
                    self.reply("Invalid; state={}".format(self.state))
            elif "STOP" in message:
                # message: "STOP:<timestamp>"
                if self.state == RecordingServerState.RECORDING:
                    timestamp = float(message.split(":")[1])
                    self.stop(timestamp)
                else:
                    self.reply("Invalid; state={}".format(self.state))
            elif "QUIT" in message:
                self.quit()
                break
            else:
                self.reply("Unknown command: {}".format(message))

    def _create_recorders(self):
        self.traj_dir = self.directory / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.recs = [
            Recorder(core.Camera(0), self.traj_dir / "cam_0", freq_limit=30),
            Recorder(core.Camera(1), self.traj_dir / "cam_1", freq_limit=30),
            Recorder(
                core.RobotStateStream(self.robot_ip),
                self.traj_dir / "robot_state.pkl",
                freq_limit=30,
            ),
            Recorder(
                core.GripperStateStream(self.robot_ip),
                self.traj_dir / "gripper_state.pkl",
                freq_limit=10,
            ),
        ]
        # add to all_recs to join at the end
        self.all_recs.extend(self.recs)

    def roll(self):
        self._create_recorders()
        for elem in self.recs:
            elem.start()
        self.state = RecordingServerState.ROLLING
        self.reply("Rolling")

    def start(self, timestamp: float):
        self.state = RecordingServerState.RECORDING
        self.reply(f"Start timestamp: {timestamp}")
        self.start_time = timestamp

    def stop(self, timestamp: float):
        self.reply(f"Stop timestamp: {timestamp}")
        self.stop_time = timestamp
        time.sleep(1)  # to make sure streams stop after stop_time
        with open(self.traj_dir / "start_stop.txt", "w") as f:
            f.write(f"{self.start_time}\n{self.stop_time}\n")

        for elem in self.recs:
            elem.stop()
        self.state = RecordingServerState.IDLE

    def quit(self):
        self.reply("Quitting")
        self.state = RecordingServerState.HALTED
        self.socket.close()
        self.context.term()
        for elem in self.all_recs:
            if not elem.stop_event.is_set():
                logging.warning("Stopping a running recorder: {}".format(elem))
                elem.stop()
        for elem in self.all_recs:
            elem.join()


class RecordingClient:
    def __init__(self, ip_address, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip_address}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # receive timeout (ms)
        self.socket.send_string("PING")
        message = self.socket.recv_string()
        logging.info(message)

        self.server_state = RecordingServerState.IDLE

    def request(self, req_msg: str, log=True):
        if log:
            logging.info(f">> {req_msg}")
        self.socket.send_string(req_msg)
        rep_msg = self.socket.recv_string()
        if log:
            logging.info(f"<< {rep_msg}")
        return rep_msg

    def roll(self):
        msg = self.request("ROLL")
        self.server_state = RecordingServerState.ROLLING

    def start(self, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        msg = self.request(f"START:{timestamp}")
        self.server_state = RecordingServerState.RECORDING

    def stop(self, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        msg = self.request(f"STOP:{timestamp}")
        self.server_state = RecordingServerState.IDLE

    def cycle_state(self):
        if self.server_state == RecordingServerState.IDLE:
            self.roll()
        elif self.server_state == RecordingServerState.ROLLING:
            self.start()
        elif self.server_state == RecordingServerState.RECORDING:
            self.stop()

    def quit(self):
        msg = self.request("QUIT")
        self.server_state = RecordingServerState.HALTED
