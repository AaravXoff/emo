"""Two MPU6050 sensors on the Pi's I2C bus (addresses 0x68 and 0x69)."""
from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


def _signed(raw: int) -> int:
    return raw - 65536 if raw > 32767 else raw


class MPU6050Array:
    """Read acceleration and gyro values from multiple MPU6050 devices."""

    PWR_MGMT_1 = 0x6B
    DATA_START = 0x3B

    def __init__(self, addresses=(0x68, 0x69), status=None, interval: float = 0.10) -> None:
        self.addresses = tuple(int(address) for address in addresses)
        self.status = status
        self.interval = max(0.05, float(interval))
        self._bus = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._latest: dict[str, Any] = {}
        self._enabled_addresses: list[int] = []

    def start(self) -> tuple[bool, str]:
        try:
            from smbus2 import SMBus
            self._bus = SMBus(1)
            for address in self.addresses:
                try:
                    self._bus.write_byte_data(address, self.PWR_MGMT_1, 0x00)
                    self._enabled_addresses.append(address)
                except OSError as exc:
                    log.warning("MPU6050 at 0x%02X is unavailable: %s", address, exc)
            if not self._enabled_addresses:
                self.stop()
                return False, "No MPU6050 found; enable I2C and check AD0 addresses 0x68/0x69"
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="EMO-MPU6050")
            self._thread.start()
            addrs = ", ".join(f"0x{address:02X}" for address in self._enabled_addresses)
            return True, f"MPU6050 active at {addrs}"
        except Exception as exc:
            self.stop()
            return False, f"MPU6050 unavailable: {exc}"

    def read_sensor(self, address: int) -> dict[str, float]:
        if self._bus is None:
            raise RuntimeError("I2C bus is not open")
        data = self._bus.read_i2c_block_data(address, self.DATA_START, 14)
        values = [_signed((data[index] << 8) | data[index + 1]) for index in range(0, 14, 2)]
        ax, ay, az, _temperature, gx, gy, gz = values
        return {
            "accel_x_g": ax / 16384.0,
            "accel_y_g": ay / 16384.0,
            "accel_z_g": az / 16384.0,
            "gyro_x_dps": gx / 131.0,
            "gyro_y_dps": gy / 131.0,
            "gyro_z_dps": gz / 131.0,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample: dict[str, Any] = {}
            for address in self._enabled_addresses:
                try:
                    sample[f"0x{address:02X}"] = self.read_sensor(address)
                except Exception as exc:
                    log.debug("MPU6050 read failed for 0x%02X: %s", address, exc)
            with self._lock:
                self._latest = sample
            if self.status is not None:
                self.status.set_imu(sample)
            self._stop.wait(self.interval)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._thread = None
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        self._enabled_addresses.clear()
