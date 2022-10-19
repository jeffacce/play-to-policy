import time
import concurrent
import polymetis
from typing import Tuple, Optional


class GripperController:
    def __init__(
        self,
        ip_address: str,
        speed: float = 0.2,
        force: float = 0.1,
        timeout: Optional[float] = None,  # seconds; None means no timeout
    ):
        self.gripper = polymetis.GripperInterface(ip_address=ip_address)
        self.max_width = self.gripper.get_state().max_width
        self.speed = speed
        self.force = force
        self.timeout = timeout
        self.pool = concurrent.futures.ThreadPoolExecutor(1)
        self.gripper.goto(self.max_width, self.speed, self.force)
        self.closed = False
        self.moving = False

    def grasp(self, blocking: bool = False):
        if not self.closed and not self.moving:
            future = self.pool.submit(self.gripper.grasp, self.speed, self.force)
            self.moving = True

            def callback(future):
                self.closed = True
                self.moving = False

            future.add_done_callback(callback)
        if blocking:
            if self.timeout is None:
                while self.moving:
                    pass
            else:
                tstart = time.time()
                while self.moving and time.time() - tstart < self.timeout:
                    pass

    def open(self, blocking: bool = False):
        if self.closed and not self.moving:
            future = self.pool.submit(
                self.gripper.goto, self.max_width, self.speed, self.force
            )
            self.moving = True

            def callback(future):
                self.closed = False
                self.moving = False

            future.add_done_callback(callback)
        if blocking:
            if self.timeout is None:
                while self.moving:
                    pass
            else:
                tstart = time.time()
                while self.moving and time.time() - tstart < self.timeout:
                    pass

    def get_state(self) -> Tuple[bool, bool]:
        """
        Returns:
            closed: bool
            moving: bool
        """
        return self.closed, self.moving
