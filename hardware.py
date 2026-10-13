"""Optional GPIO touch input, one-servo expression feedback, and Pi thermal monitoring."""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class HardwareController:
    def __init__(self, touch_gpio: int = 0, servo_gpio: int = 0, status=None) -> None:
        self.touch_gpio = touch_gpio
        self.servo_gpio = servo_gpio
        self.status = status
        self.button = None
        self.servo = None

    def start(self) -> tuple[bool, str]:
        enabled = []
        errors = []
        if not self.touch_gpio and not self.servo_gpio:
            return True, "GPIO disabled (configure TOUCH_GPIO/SERVO_GPIO after checking wiring)"
        try:
            from gpiozero import Button, AngularServo
            if self.touch_gpio:
                self.button = Button(self.touch_gpio, pull_up=True, bounce_time=0.12)
                self.button.when_pressed = self._touch_pressed
                enabled.append(f"touch GPIO {self.touch_gpio}")
            if self.servo_gpio:
                self.servo = AngularServo(
                    self.servo_gpio, min_angle=-90, max_angle=90,
                    min_pulse_width=0.0005, max_pulse_width=0.0025,
                    initial_angle=0,
                )
                enabled.append(f"servo GPIO {self.servo_gpio}")
        except Exception as exc:
            errors.append(str(exc))
            log.warning("GPIO setup failed: %s", exc)
        if errors:
            return False, "; ".join(errors)
        return True, "Enabled " + ", ".join(enabled)

    def _touch_pressed(self) -> None:
        if self.status is not None:
            with self.status.lock:
                self.status.last_touch = time.monotonic()

    def set_expression(self, expression: str) -> None:
        if self.servo is None:
            return
        angles = {
            "idle": 0, "listening": -12, "thinking": 10, "happy": 30,
            "excited": 45, "dance": 45, "sad": -25, "angry": -35,
            "surprised": 20, "overheat": 0, "talking": 8, "curious": 18,
        }
        try:
            self.servo.angle = angles.get(expression, 0)
        except Exception as exc:
            log.debug("Servo position failed: %s", exc)

    def close(self) -> None:
        for device in (self.button, self.servo):
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass


class ThermalMonitor:
    def __init__(self, status, warning_c: float = 75.0, lockout_c: float = 82.0,
                 resume_c: float = 72.0, interval: float = 4.0) -> None:
        self.status = status
        self.warning_c = warning_c
        self.lockout_c = lockout_c
        self.resume_c = resume_c
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self._locked_out = False

    @staticmethod
    def read_temperature() -> float:
        candidates = (
            Path("/sys/class/thermal/thermal_zone0/temp"),
            Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
        )
        for path in candidates:
            try:
                value = float(path.read_text(encoding="ascii").strip())
                return value / 1000.0 if value > 200 else value
            except Exception:
                pass
        return 0.0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True, name="EMO-Thermal")
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            temperature = self.read_temperature()
            if temperature >= self.lockout_c:
                self._locked_out = True
            elif self._locked_out and temperature <= self.resume_c:
                self._locked_out = False
            warning = temperature >= self.warning_c
            self.status.set_thermal(temperature, warning, self._locked_out)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
