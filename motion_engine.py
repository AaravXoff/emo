"""Safe, named four-servo poses and A-Z movement routines for EMO."""
from __future__ import annotations

import logging
import queue
import threading

log = logging.getLogger(__name__)

HOME = (0, 0, 0, 0)
# Waypoint = (head yaw, head pitch, left arm, right arm), pause seconds.
MOTIONS = {
    "A": ("wave", [(HOME, .15), ((0, 0, -15, 42), .25), ((0, 0, -15, 15), .20), ((0, 0, -15, 42), .25), (HOME, .15)]),
    "B": ("nod", [(HOME, .15), ((0, 22, 0, 0), .30), ((0, -10, 0, 0), .30), (HOME, .15)]),
    "C": ("look left", [((-35, 0, 0, 0), .35), (HOME, .15)]),
    "D": ("look right", [((35, 0, 0, 0), .35), (HOME, .15)]),
    "E": ("bow", [((0, 12, 0, 0), .18), ((0, -28, 0, 0), .5), (HOME, .18)]),
    "F": ("celebrate", [((0, -8, -40, 40), .25), ((0, -8, -20, 20), .20), ((0, -8, -40, 40), .25), (HOME, .2)]),
    "G": ("shy", [((0, -12, -12, -12), .35), ((-10, -7, -12, -12), .25), (HOME, .2)]),
    "H": ("curious", [((20, -5, 0, 10), .4), ((-10, -5, 0, -10), .4), (HOME, .15)]),
    "I": ("stretch", [((0, -5, -48, 48), .45), (HOME, .2)]),
    "J": ("dance", [((-22, -5, -32, 32), .22), ((22, 5, 32, -32), .22), ((-22, 5, -32, 32), .22), (HOME, .15)]),
    "K": ("greet", [((0, 0, -12, 38), .3), ((0, 0, -12, 14), .2), (HOME, .15)]),
    "L": ("point left", [((-28, 0, -36, 5), .4), (HOME, .15)]),
    "M": ("point right", [((28, 0, 5, 36), .4), (HOME, .15)]),
    "N": ("yes", [((0, 14, 0, 0), .22), ((0, -8, 0, 0), .22), ((0, 14, 0, 0), .22), (HOME, .15)]),
    "O": ("no", [((-16, 0, 0, 0), .22), ((16, 0, 0, 0), .22), ((-16, 0, 0, 0), .22), (HOME, .15)]),
    "P": ("sleepy", [((0, -18, -8, -8), .55), (HOME, .25)]),
    "Q": ("surprised", [((0, 14, 28, -28), .35), (HOME, .2)]),
    "R": ("scan", [((-32, 0, 0, 0), .3), ((0, -4, 0, 0), .2), ((32, 0, 0, 0), .3), (HOME, .2)]),
    "S": ("march", [((-8, 2, 25, -25), .18), ((8, -2, -25, 25), .18), ((-8, 2, 25, -25), .18), (HOME, .15)]),
    "T": ("flap", [((0, 0, -35, 35), .25), ((0, 0, 0, 0), .18), ((0, 0, -35, 35), .25), (HOME, .15)]),
    "U": ("shrug", [((0, -4, -20, 20), .2), ((0, 8, -32, 32), .4), (HOME, .15)]),
    "V": ("victory", [((0, -5, -45, 45), .45), (HOME, .15)]),
    "W": ("double wave", [((0, 0, -38, 38), .25), ((0, 0, -15, 15), .18), ((0, 0, -38, 38), .25), (HOME, .15)]),
    "X": ("cross arms", [((0, 0, 22, 22), .4), (HOME, .18)]),
    "Y": ("look up", [((0, 28, 0, 0), .35), (HOME, .15)]),
    "Z": ("reset", [(HOME, .25)]),
}
ALIASES = {name.casefold(): key for key, (name, _steps) in MOTIONS.items()}


class MotionEngine:
    def __init__(self, hardware, stop_event=None) -> None:
        self.hardware = hardware
        self.stop_event = stop_event or threading.Event()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=12)
        self._worker = None

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True, name="EMO-Motion")
        self._worker.start()

    def run_async(self, name: str) -> bool:
        key = (name or "").strip().upper()
        if key not in MOTIONS:
            key = ALIASES.get((name or "").strip().casefold(), "")
        if key not in MOTIONS:
            return False
        try:
            self._queue.put_nowait(key)
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                key = self._queue.get(timeout=.25)
            except queue.Empty:
                continue
            if key is None:
                self._queue.task_done()
                break
            _name, steps = MOTIONS[key]
            for pose, pause in steps:
                if self.stop_event.is_set():
                    break
                self.hardware.set_pose(pose)
                self.stop_event.wait(pause)
            self.hardware.set_expression("idle")
            self._queue.task_done()

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=2)
