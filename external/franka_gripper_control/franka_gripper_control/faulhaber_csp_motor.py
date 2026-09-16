#!/usr/bin/env python3
"""Motor-side lifecycle wrapper for FAULHABER gripper CSP control.

This module deliberately contains no ROS dependencies. It contains the tested
CANopen primitives and presents a small API to the ROS trajectory node:

    start() -> command_width_mm() / read_feedback() -> stop()

The caller owns scheduling.  Each command stages RPDO1 and emits SYNC; TPDO4
then supplies statusword, actual position, and motor current.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time

import can

# FAULHABER CANopen primitives are intentionally defined in this file so the
# ROS node has exactly one standalone motor-side dependency.
class DriveError(RuntimeError):
    pass


class FaulhaberDrive:
    def __init__(self, bus: can.BusABC, node_id: int = 1, timeout: float = 1.0):
        self.bus = bus
        self.node_id = node_id
        self.timeout = timeout
        self.sdo_tx = 0x600 + node_id
        self.sdo_rx = 0x580 + node_id
        self.heartbeat = 0x700 + node_id

    def send(self, arbitration_id: int, data: bytes = b"") -> None:
        self.bus.send(
            can.Message(
                arbitration_id=arbitration_id,
                data=data,
                is_extended_id=False,
            )
        )

    def _wait_sdo(self, index: int, subindex: int) -> bytes:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            message = self.bus.recv(deadline - time.monotonic())
            if message is None:
                break

            # A boot-up during a command means that power/controller reset.
            if (
                message.arbitration_id == self.heartbeat
                and len(message.data) == 1
                and message.data[0] == 0
            ):
                raise DriveError("Controller rebooted during communication")

            if message.arbitration_id != self.sdo_rx or len(message.data) != 8:
                continue

            data = bytes(message.data)
            response_index = data[1] | (data[2] << 8)
            if response_index != index or data[3] != subindex:
                continue

            if data[0] == 0x80:
                abort = int.from_bytes(data[4:8], "little")
                raise DriveError(
                    f"SDO abort 0x{abort:08X} for 0x{index:04X}:{subindex:02X}"
                )
            return data

        raise DriveError(f"No SDO response for 0x{index:04X}:{subindex:02X}")

    def read(self, index: int, subindex: int = 0, *, signed: bool = False) -> int:
        request = bytes((0x40, index & 0xFF, index >> 8, subindex, 0, 0, 0, 0))
        self.send(self.sdo_tx, request)
        response = self._wait_sdo(index, subindex)

        sizes = {0x4F: 1, 0x4B: 2, 0x47: 3, 0x43: 4}
        size = sizes.get(response[0])
        if size is None:
            raise DriveError(f"Unexpected SDO upload response 0x{response[0]:02X}")
        return int.from_bytes(response[4 : 4 + size], "little", signed=signed)

    def write_u8(self, index: int, value: int, subindex: int = 0) -> None:
        self._write(index, subindex, 0x2F, struct.pack("<B", value))

    def write_u16(self, index: int, value: int, subindex: int = 0) -> None:
        self._write(index, subindex, 0x2B, struct.pack("<H", value))

    def write_i32(self, index: int, value: int, subindex: int = 0) -> None:
        self._write(index, subindex, 0x23, struct.pack("<i", value))

    def write_u32(self, index: int, value: int, subindex: int = 0) -> None:
        self._write(index, subindex, 0x23, struct.pack("<I", value))

    def _write(self, index: int, subindex: int, command: int, payload: bytes) -> None:
        data = bytearray(8)
        data[0] = command
        data[1] = index & 0xFF
        data[2] = index >> 8
        data[3] = subindex
        data[4 : 4 + len(payload)] = payload
        self.send(self.sdo_tx, bytes(data))
        response = self._wait_sdo(index, subindex)
        if response[0] != 0x60:
            raise DriveError(f"Unexpected SDO download response 0x{response[0]:02X}")

    def statusword(self) -> int:
        return self.read(0x6041)

    @staticmethod
    def state(statusword: int) -> int:
        return statusword & 0x006F

    def require_state(self, expected: int, label: str) -> int:
        status = self.statusword()
        if status & 0x0008:
            raise DriveError(f"Drive fault, statusword 0x{status:04X}")
        if self.state(status) != expected:
            raise DriveError(
                f"Expected {label} (0x{expected:04X}), got statusword 0x{status:04X}"
            )
        print(f"{label}: statusword 0x{status:04X}")
        return status

    def enable(self) -> None:
        # NMT Start Remote Node for this node.
        self.send(0x000, bytes((0x01, self.node_id)))
        time.sleep(0.05)

        # Explicitly select Profile Position mode.
        self.write_u8(0x6060, 1)
        if self.read(0x6061, signed=True) != 1:
            raise DriveError("Profile Position mode did not become active")

        for controlword, expected, label in (
            (0x0006, 0x0021, "Ready to Switch On"),
            (0x0007, 0x0023, "Switched On"),
            (0x000F, 0x0027, "Operation Enabled"),
        ):
            self.write_u16(0x6040, controlword)
            self.require_state(expected, label)

    def move_relative(self, increments: int, move_timeout: float) -> tuple[int, int]:
        initial = self.read(0x6064, signed=True)
        print(f"Initial position: {initial} increments")

        self.write_i32(0x607A, increments)

        # Relative + change immediately + rising edge of New Setpoint.
        self.write_u16(0x6040, 0x007F)

        deadline = time.monotonic() + move_timeout
        acknowledged = False
        try:
            while time.monotonic() < deadline:
                status = self.statusword()
                if status & 0x0008:
                    raise DriveError(f"Drive fault during movement: 0x{status:04X}")
                if self.state(status) != 0x0027:
                    raise DriveError(f"Drive left Operation Enabled: 0x{status:04X}")
                acknowledged |= bool(status & (1 << 12))
                if acknowledged and status & (1 << 10):
                    final = self.read(0x6064, signed=True)
                    return initial, final
                time.sleep(0.05)
        finally:
            # Clear New Setpoint while retaining relative/immediate selection.
            try:
                self.write_u16(0x6040, 0x006F)
            except Exception:
                pass

        raise DriveError("Timed out waiting for the target position")

    def home_current_position(self, homing_timeout: float) -> int:
        """Run CiA 402 homing method 35: define current position as zero."""
        self.write_u8(0x6060, 6)
        if self.read(0x6061, signed=True) != 6:
            raise DriveError("Homing mode did not become active")

        self.write_u8(0x6098, 35)
        self.write_u16(0x6040, 0x001F)

        deadline = time.monotonic() + homing_timeout
        try:
            while time.monotonic() < deadline:
                status = self.statusword()
                if status & 0x0008:
                    raise DriveError(f"Drive fault during homing: 0x{status:04X}")
                if self.state(status) != 0x0027:
                    raise DriveError(f"Drive left Operation Enabled: 0x{status:04X}")
                if status & (1 << 13):
                    raise DriveError(f"Homing error, statusword 0x{status:04X}")
                if status & (1 << 12) and status & (1 << 10):
                    break
                time.sleep(0.05)
            else:
                raise DriveError("Timed out waiting for homing to complete")
        finally:
            # Clear Homing Operation Start.
            try:
                self.write_u16(0x6040, 0x000F)
            except Exception:
                pass

        position = self.read(0x6064, signed=True)
        self.write_u8(0x6060, 1)
        if self.read(0x6061, signed=True) != 1:
            raise DriveError("Profile Position mode did not become active after homing")
        return position

    def _approach_stop(
        self,
        direction: int,
        label: str,
        speed_rpm: int,
        current_threshold_ma: int,
        stall_duration: float,
        max_travel: int,
        approach_timeout: float,
        poll_interval: float,
        progress_interval: float,
        stall_velocity_rpm: int,
        stall_position_span: int,
    ) -> tuple[int, int]:
        """Approach one stop and detect sustained current plus stopped motion."""
        if direction not in {-1, 1}:
            raise ValueError("direction must be -1 or +1")
        speed_rpm = abs(speed_rpm) * direction

        self.write_u8(0x6060, 3)  # Profile Velocity
        if self.read(0x6061, signed=True) != 3:
            raise DriveError("Profile Velocity mode did not become active")

        self.write_i32(0x60FF, 0)
        start_position = self.read(0x6064, signed=True)
        self.write_i32(0x60FF, speed_rpm)

        deadline = time.monotonic() + approach_timeout
        started_at = time.monotonic()
        stall_since: float | None = None
        stall_anchor_position: int | None = None
        last_progress = 0.0
        loop_count = 0

        try:
            while time.monotonic() < deadline:
                now = time.monotonic()
                # Status is sampled less often to keep SDO traffic modest.
                if loop_count % 5 == 0:
                    status = self.statusword()
                    if status & 0x0008:
                        raise DriveError(f"Drive fault during stop approach: 0x{status:04X}")
                    if self.state(status) != 0x0027:
                        raise DriveError(f"Drive left Operation Enabled: 0x{status:04X}")

                position = self.read(0x6064, signed=True)
                velocity = self.read(0x606C, signed=True)
                current = self.read(0x6078, signed=True)
                signed_travel = (position - start_position) * direction

                if signed_travel < -50:
                    raise DriveError(
                        f"Position moved opposite the expected {label} direction"
                    )
                if signed_travel > max_travel:
                    raise DriveError(
                        f"{label.capitalize()} stop not detected within "
                        f"{max_travel} increments"
                    )

                if now - last_progress >= progress_interval:
                    print(
                        f"  {label}: travel={signed_travel}, position={position}, "
                        f"velocity={velocity} rpm, current={current} mA"
                    )
                    last_progress = now

                low_velocity = abs(velocity) <= stall_velocity_rpm
                loaded = abs(current) >= current_threshold_ma

                # Ignore the first 300 ms so acceleration is not mistaken for contact.
                if now - started_at >= 0.3 and low_velocity and loaded:
                    if stall_since is None:
                        stall_since = now
                        stall_anchor_position = position
                    elif abs(position - stall_anchor_position) > stall_position_span:
                        # It is still making meaningful progress under load. Start a
                        # fresh stationary interval from the new position.
                        stall_since = now
                        stall_anchor_position = position
                    if now - stall_since >= stall_duration:
                        return position, current
                else:
                    stall_since = None
                    stall_anchor_position = None

                loop_count += 1
                time.sleep(poll_interval)
        finally:
            # Stop velocity even if detection, communication, or validation fails.
            try:
                self.write_i32(0x60FF, 0)
            except Exception:
                pass

        raise DriveError(f"Timed out before detecting the {label} mechanical stop")

    def calibrate_hard_stops(
        self,
        *,
        current_limit_ma: int,
        current_threshold_ma: int,
        first_speed_rpm: int,
        second_speed_rpm: int,
        backoff_increments: int,
        stall_duration: float,
        max_open_travel: int,
        max_close_travel: int,
        approach_timeout: float,
        poll_interval: float,
        progress_interval: float,
        repeatability_tolerance: int,
        endpoint_margin: int,
        stall_velocity_rpm: int,
        stall_position_span: int,
    ) -> dict[str, int]:
        """Find both hard stops twice and return raw and safe absolute limits."""
        original = {
            "continuous_current": self.read(0x2333, 1),
            "peak_current": self.read(0x2333, 2),
            "acceleration": self.read(0x6083),
            "deceleration": self.read(0x6084),
        }

        # Never increase a valid stored current limit during calibration.
        applied_continuous = min(current_limit_ma, original["continuous_current"])
        applied_peak = min(current_limit_ma, original["peak_current"])
        applied_limit = min(applied_continuous, applied_peak)
        if applied_limit <= 0:
            raise DriveError("Controller reported an invalid stored current limit")
        if current_threshold_ma >= applied_limit:
            raise DriveError(
                f"Stall threshold {current_threshold_ma} mA must be below the "
                f"applied current limit {applied_limit} mA"
            )

        print(
            f"Temporary homing limits: current={applied_limit} mA, "
            "acceleration=100 1/s^2"
        )

        try:
            self.write_u16(0x2333, applied_limit, 1)
            self.write_u16(0x2333, applied_limit, 2)
            self.write_u32(0x6083, 100)
            self.write_u32(0x6084, 100)

            position, current = self._approach_stop(
                1,
                "opening",
                first_speed_rpm,
                current_threshold_ma,
                stall_duration,
                max_open_travel,
                approach_timeout,
                poll_interval,
                progress_interval,
                stall_velocity_rpm,
                stall_position_span,
            )
            print(f"First open-stop detection: position={position}, current={current} mA")

            # Back away in the known negative/closing direction.
            self.write_u8(0x6060, 1)
            if self.read(0x6061, signed=True) != 1:
                raise DriveError("Profile Position mode did not become active")
            initial, final = self.move_relative(-abs(backoff_increments), approach_timeout)
            print(f"Backed away from stop: {initial} -> {final}")

            position, current = self._approach_stop(
                1,
                "opening",
                second_speed_rpm,
                current_threshold_ma,
                stall_duration,
                max_open_travel,
                approach_timeout,
                poll_interval,
                progress_interval,
                stall_velocity_rpm,
                stall_position_span,
            )
            print(f"Second open-stop detection: position={position}, current={current} mA")

            zero = self.home_current_position(approach_timeout)
            print(f"Fully-open home established: position={zero}")

            # First negative/closing pass from the newly established zero.
            closed_first, current = self._approach_stop(
                -1,
                "closing",
                first_speed_rpm,
                current_threshold_ma,
                stall_duration,
                max_close_travel,
                approach_timeout,
                poll_interval,
                progress_interval,
                stall_velocity_rpm,
                stall_position_span,
            )
            print(
                f"First closed-stop detection: position={closed_first}, "
                f"current={current} mA"
            )

            # Back away in the positive/opening direction and approach again slowly.
            self.write_u8(0x6060, 1)
            if self.read(0x6061, signed=True) != 1:
                raise DriveError("Profile Position mode did not become active")
            initial, final = self.move_relative(abs(backoff_increments), approach_timeout)
            print(f"Backed away from closed stop: {initial} -> {final}")

            closed_second, current = self._approach_stop(
                -1,
                "closing",
                second_speed_rpm,
                current_threshold_ma,
                stall_duration,
                max_close_travel,
                approach_timeout,
                poll_interval,
                progress_interval,
                stall_velocity_rpm,
                stall_position_span,
            )
            print(
                f"Second closed-stop detection: position={closed_second}, "
                f"current={current} mA"
            )

            repeatability = abs(closed_second - closed_first)
            if repeatability > repeatability_tolerance:
                raise DriveError(
                    f"Closed-stop detections differ by {repeatability} increments; "
                    f"limit is {repeatability_tolerance}"
                )

            raw_closed = int(round((closed_first + closed_second) / 2.0))
            safe_min = raw_closed + abs(endpoint_margin)
            safe_max = -abs(endpoint_margin)
            if safe_min >= safe_max:
                raise DriveError("Calibrated travel is smaller than the endpoint margins")

            # Leave the mechanism off both hard stops.
            self.write_u8(0x6060, 1)
            if self.read(0x6061, signed=True) != 1:
                raise DriveError("Profile Position mode did not become active")
            self.move_absolute(safe_min, approach_timeout)

            return {
                "open_stop": 0,
                "closed_stop_first": closed_first,
                "closed_stop_second": closed_second,
                "closed_stop": raw_closed,
                "safe_min": safe_min,
                "safe_max": safe_max,
                "endpoint_margin": abs(endpoint_margin),
                "repeatability": repeatability,
            }
        finally:
            # Best effort: stop motion and restore every changed parameter.
            try:
                self.write_i32(0x60FF, 0)
            except Exception:
                pass
            for writer, index, value, subindex in (
                (self.write_u16, 0x2333, original["continuous_current"], 1),
                (self.write_u16, 0x2333, original["peak_current"], 2),
                (self.write_u32, 0x6083, original["acceleration"], 0),
                (self.write_u32, 0x6084, original["deceleration"], 0),
            ):
                try:
                    writer(index, value, subindex)
                except Exception:
                    pass
            print("Original current and motion limits restored")

    def move_absolute(self, target: int, move_timeout: float) -> tuple[int, int]:
        """Move to an absolute CiA 402 position coordinate."""
        initial = self.read(0x6064, signed=True)
        self.write_i32(0x607A, target)

        # Absolute + change immediately + rising edge of New Setpoint.
        # Bit 6 is clear, distinguishing this from relative controlword 0x007F.
        self.write_u16(0x6040, 0x003F)

        deadline = time.monotonic() + move_timeout
        acknowledged = False
        try:
            while time.monotonic() < deadline:
                status = self.statusword()
                if status & 0x0008:
                    raise DriveError(f"Drive fault during movement: 0x{status:04X}")
                if self.state(status) != 0x0027:
                    raise DriveError(f"Drive left Operation Enabled: 0x{status:04X}")
                acknowledged |= bool(status & (1 << 12))
                if acknowledged and status & (1 << 10):
                    final = self.read(0x6064, signed=True)
                    return initial, final
                time.sleep(0.05)
        finally:
            # Clear New Setpoint; remain in absolute/immediate operation.
            try:
                self.write_u16(0x6040, 0x002F)
            except Exception:
                pass

        raise DriveError("Timed out waiting for the absolute target")

    def snapshot_rpdo1(self) -> dict[str, int | list[int]]:
        """Read the RPDO1 communication and mapping parameters for restoration."""
        count = self.read(0x1600, 0)
        if count > 8:
            raise DriveError(f"Unexpected RPDO1 mapping count {count}")
        return {
            "cob_id": self.read(0x1400, 1),
            "transmission_type": self.read(0x1400, 2),
            "mapping": [self.read(0x1600, i) for i in range(1, count + 1)],
        }

    def configure_csp_rpdo1(self) -> dict[str, int | list[int]]:
        """Temporarily map RPDO1 to controlword + target position, synchronous."""
        saved = self.snapshot_rpdo1()
        try:
            disabled_cob_id = int(saved["cob_id"]) | 0x80000000
            self.write_u32(0x1400, disabled_cob_id, 1)
            self.write_u8(0x1600, 0, 0)
            self.write_u32(0x1600, 0x60400010, 1)  # controlword, 16 bits
            self.write_u32(0x1600, 0x607A0020, 2)  # target position, 32 bits
            self.write_u8(0x1600, 2, 0)
            self.write_u8(0x1400, 1, 2)  # synchronous, every SYNC
            self.write_u32(0x1400, 0x200 + self.node_id, 1)
        except Exception:
            try:
                self.restore_rpdo1(saved)
            except Exception:
                pass
            raise
        return saved

    def restore_rpdo1(self, saved: dict[str, int | list[int]]) -> None:
        """Best-effort restoration of the RPDO1 configuration."""
        original_cob_id = int(saved["cob_id"])
        self.write_u32(0x1400, original_cob_id | 0x80000000, 1)
        self.write_u8(0x1600, 0, 0)
        for subindex, entry in enumerate(saved["mapping"], start=1):
            self.write_u32(0x1600, int(entry), subindex)
        self.write_u8(0x1600, len(saved["mapping"]), 0)
        self.write_u8(0x1400, int(saved["transmission_type"]), 2)
        self.write_u32(0x1400, original_cob_id, 1)

    def snapshot_tpdo4(self) -> dict[str, int | list[int]]:
        """Read TPDO4 communication and mapping parameters for restoration."""
        count = self.read(0x1A03, 0)
        if count > 8:
            raise DriveError(f"Unexpected TPDO4 mapping count {count}")
        return {
            "cob_id": self.read(0x1803, 1),
            "transmission_type": self.read(0x1803, 2),
            "mapping": [self.read(0x1A03, i) for i in range(1, count + 1)],
        }

    def configure_csp_tpdo4(self) -> dict[str, int | list[int]]:
        """Map synchronous TPDO4 to status, actual position, and current."""
        saved = self.snapshot_tpdo4()
        try:
            disabled_cob_id = int(saved["cob_id"]) | 0x80000000
            self.write_u32(0x1803, disabled_cob_id, 1)
            self.write_u8(0x1A03, 0, 0)
            self.write_u32(0x1A03, 0x60410010, 1)  # statusword, 16 bits
            self.write_u32(0x1A03, 0x60640020, 2)  # actual position, 32 bits
            self.write_u32(0x1A03, 0x60780010, 3)  # actual current, 16 bits
            self.write_u8(0x1A03, 3, 0)
            self.write_u8(0x1803, 1, 2)  # transmit after every SYNC
            self.write_u32(0x1803, 0x480 + self.node_id, 1)
        except Exception:
            try:
                self.restore_tpdo4(saved)
            except Exception:
                pass
            raise
        return saved

    def restore_tpdo4(self, saved: dict[str, int | list[int]]) -> None:
        """Best-effort restoration of the TPDO4 configuration."""
        original_cob_id = int(saved["cob_id"])
        self.write_u32(0x1803, original_cob_id | 0x80000000, 1)
        self.write_u8(0x1A03, 0, 0)
        for subindex, entry in enumerate(saved["mapping"], start=1):
            self.write_u32(0x1A03, int(entry), subindex)
        self.write_u8(0x1A03, len(saved["mapping"]), 0)
        self.write_u8(0x1803, int(saved["transmission_type"]), 2)
        self.write_u32(0x1803, original_cob_id, 1)

    def enter_csp(
        self,
    ) -> tuple[
        dict[str, int | list[int]], dict[str, int | list[int]], int
    ]:
        """Initialize CSP, configure command/feedback PDOs, and re-enable."""
        self.write_u16(0x6040, 0x0000)
        time.sleep(0.05)
        current_position = self.read(0x6064, signed=True)
        self.write_i32(0x607A, current_position)
        self.write_u8(0x6060, 8)
        if self.read(0x6061, signed=True) != 8:
            raise DriveError("Cyclic Synchronous Position mode did not become active")

        saved_rpdo1 = self.configure_csp_rpdo1()
        saved_tpdo4: dict[str, int | list[int]] | None = None
        try:
            saved_tpdo4 = self.configure_csp_tpdo4()
            for controlword, expected, label in (
                (0x0006, 0x0021, "CSP Ready to Switch On"),
                (0x0007, 0x0023, "CSP Switched On"),
                (0x000F, 0x0027, "CSP Operation Enabled"),
            ):
                self.write_u16(0x6040, controlword)
                self.require_state(expected, label)
        except Exception:
            try:
                self.write_u16(0x6040, 0x0000)
                if saved_tpdo4 is not None:
                    self.restore_tpdo4(saved_tpdo4)
                self.restore_rpdo1(saved_rpdo1)
                self.write_u8(0x6060, 1)
            except Exception:
                pass
            raise
        return saved_rpdo1, saved_tpdo4, current_position

    def send_csp_target(self, raw_target: int) -> None:
        """Stage one synchronous RPDO1 target and apply it with CANopen SYNC."""
        payload = struct.pack("<Hi", 0x000F, raw_target)
        self.send(0x200 + self.node_id, payload)
        self.send(0x080)

    def read_csp_tpdo4(self, timeout: float) -> tuple[int, int, int]:
        """Receive one synchronous TPDO4: statusword, position, current."""
        deadline = time.monotonic() + timeout
        cob_id = 0x480 + self.node_id
        while time.monotonic() < deadline:
            message = self.bus.recv(max(0.0, deadline - time.monotonic()))
            if message is None:
                break
            if (
                message.arbitration_id == self.heartbeat
                and len(message.data) == 1
                and message.data[0] == 0
            ):
                raise DriveError("Controller rebooted during CSP")
            if message.arbitration_id == cob_id and len(message.data) == 8:
                return struct.unpack("<Hih", bytes(message.data))
        raise DriveError(f"No synchronous TPDO4 feedback within {timeout*1000:.1f} ms")

    def quick_stop(self) -> None:
        """Request CiA 402 Quick Stop using SDO."""
        self.write_u16(0x6040, 0x0002)

    def disable(self) -> None:
        self.write_u16(0x6040, 0x0000)


def raw_to_width_mm(raw_position: int, limits: dict[str, int | float]) -> float:
    """Map raw motor coordinates to gripper opening width in millimetres."""
    raw_span = limits["safe_max"] - limits["safe_min"]
    return (raw_position - limits["safe_min"]) * limits["max_width_mm"] / raw_span


def width_mm_to_raw_position(width_mm: float, limits: dict[str, int | float]) -> int:
    """Map gripper opening width in millimetres to raw motor coordinates."""
    raw_span = limits["safe_max"] - limits["safe_min"]
    return int(round(limits["safe_min"] + width_mm * raw_span / limits["max_width_mm"]))




@dataclass(frozen=True)
class CalibrationOptions:
    current_limit_ma: int = 300
    stall_current_ma: int = 220
    first_speed_rpm: int = 40
    second_speed_rpm: int = 10
    backoff_increments: int = 100
    stall_duration: float = 0.4
    max_open_travel: int = 1_000_000
    max_close_travel: int = 2_000_000
    homing_timeout: float = 180.0
    poll_interval: float = 0.1
    progress_interval: float = 0.5
    repeatability_tolerance: int = 100
    endpoint_margin: int = 100
    stall_velocity_rpm: int = 5
    stall_position_span: int = 25
    max_width_mm: float = 81.2


@dataclass(frozen=True)
class CSPOptions:
    current_limit_ma: int = 400
    feedback_timeout_ms: float = 4.0


@dataclass(frozen=True)
class Feedback:
    statusword: int
    raw_position: int
    width_mm: float
    current_ma: int


class FaulhaberCSPMotor:
    """Own the SocketCAN bus, calibration, PDO mappings, and safe teardown."""

    def __init__(
        self,
        interface: str = "can0",
        node_id: int = 1,
        *,
        calibration: CalibrationOptions | None = None,
        csp: CSPOptions | None = None,
        calibration_path: Path | str = "faulhaber_gripper_limits.json",
        recalibrate: bool = True,
    ) -> None:
        if not 1 <= node_id <= 127:
            raise ValueError("node_id must be from 1 through 127")
        self.interface = interface
        self.node_id = node_id
        self.calibration_options = calibration or CalibrationOptions()
        self.csp_options = csp or CSPOptions()
        self.calibration_path = Path(calibration_path)
        self.recalibrate = recalibrate

        self.bus: can.BusABC | None = None
        self.drive: FaulhaberDrive | None = None
        self.limits: dict[str, int | float] | None = None
        self._saved_rpdo1: dict[str, int | list[int]] | None = None
        self._saved_tpdo4: dict[str, int | list[int]] | None = None
        self._original_mode: int | None = None
        self._original_continuous_current: int | None = None
        self._original_peak_current: int | None = None
        self._started = False

    def _require_drive(self) -> FaulhaberDrive:
        if self.drive is None:
            raise DriveError("Motor connection is not open")
        return self.drive

    def _load_calibration(self) -> dict[str, int | float]:
        record = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        required = {"safe_min", "safe_max", "max_width_mm"}
        missing = required.difference(record)
        if missing:
            raise DriveError(
                f"Calibration file lacks fields: {', '.join(sorted(missing))}"
            )
        if float(record["max_width_mm"]) <= 0:
            raise DriveError("Calibration max_width_mm must be positive")
        if int(record["safe_min"]) >= int(record["safe_max"]):
            raise DriveError("Calibration raw limits are invalid")
        return record

    def _calibrate(self) -> dict[str, int | float]:
        drive = self._require_drive()
        cfg = self.calibration_options
        limits: dict[str, int | float] = drive.calibrate_hard_stops(
            current_limit_ma=cfg.current_limit_ma,
            current_threshold_ma=cfg.stall_current_ma,
            first_speed_rpm=cfg.first_speed_rpm,
            second_speed_rpm=cfg.second_speed_rpm,
            backoff_increments=cfg.backoff_increments,
            stall_duration=cfg.stall_duration,
            max_open_travel=cfg.max_open_travel,
            max_close_travel=cfg.max_close_travel,
            approach_timeout=cfg.homing_timeout,
            poll_interval=cfg.poll_interval,
            progress_interval=cfg.progress_interval,
            repeatability_tolerance=cfg.repeatability_tolerance,
            endpoint_margin=cfg.endpoint_margin,
            stall_velocity_rpm=cfg.stall_velocity_rpm,
            stall_position_span=cfg.stall_position_span,
        )
        limits["max_width_mm"] = cfg.max_width_mm
        record = {
            **limits,
            "node_id": self.node_id,
            "interface": self.interface,
            "calibrated_unix_time": time.time(),
        }
        self.calibration_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Calibration saved to {self.calibration_path}")
        return record

    def start(self) -> Feedback:
        """Connect, calibrate/load limits, enter CSP, and return initial feedback."""
        if self._started:
            raise DriveError("Motor is already started")
        self.bus = can.Bus(interface="socketcan", channel=self.interface)
        self.drive = FaulhaberDrive(self.bus, self.node_id)
        drive = self.drive
        try:
            status = drive.statusword()
            self._original_mode = drive.read(0x6061, signed=True)
            position = drive.read(0x6064, signed=True)
            print(
                f"Connected: statusword=0x{status:04X}, "
                f"mode={self._original_mode}, position={position}"
            )
            drive.enable()
            if self.recalibrate:
                self.limits = self._calibrate()
            else:
                self.limits = self._load_calibration()
                print(f"Loaded calibration from {self.calibration_path}")

            self._original_continuous_current = drive.read(0x2333, 1)
            self._original_peak_current = drive.read(0x2333, 2)
            applied = min(
                self.csp_options.current_limit_ma,
                self._original_continuous_current,
                self._original_peak_current,
            )
            if applied <= 0:
                raise DriveError("Controller reported an invalid current limit")
            drive.write_u16(0x2333, applied, 1)
            drive.write_u16(0x2333, applied, 2)
            print(f"Temporary CSP current limit: {applied} mA")

            self._saved_rpdo1, self._saved_tpdo4, initial_raw = drive.enter_csp()
            self._started = True
            # Generate the first synchronous feedback sample while holding position.
            drive.send_csp_target(initial_raw)
            return self.read_feedback()
        except BaseException:
            self.stop(quick=True)
            raise

    def calibrate_and_restart(self) -> Feedback:
        """Leave CSP, calibrate both hard stops, save limits, and re-enter CSP."""
        self.stop(quick=False)
        previous = self.recalibrate
        self.recalibrate = True
        try:
            return self.start()
        finally:
            self.recalibrate = previous

    @property
    def started(self) -> bool:
        return self._started

    def command_raw(self, raw_target: int) -> None:
        if not self._started or self.limits is None:
            raise DriveError("CSP motor has not been started")
        safe_min = int(self.limits["safe_min"])
        safe_max = int(self.limits["safe_max"])
        if not safe_min <= raw_target <= safe_max:
            raise DriveError(
                f"Raw target {raw_target} is outside [{safe_min}, {safe_max}]"
            )
        self._require_drive().send_csp_target(raw_target)

    def command_width_mm(self, width_mm: float) -> None:
        if self.limits is None:
            raise DriveError("No calibration is available")
        maximum = float(self.limits["max_width_mm"])
        if not 0.0 <= width_mm <= maximum:
            raise DriveError(
                f"Width {width_mm:.3f} mm is outside [0, {maximum:g}] mm"
            )
        self.command_raw(width_mm_to_raw_position(width_mm, self.limits))

    def read_feedback(self) -> Feedback:
        if not self._started or self.limits is None:
            raise DriveError("CSP motor has not been started")
        status, raw, current = self._require_drive().read_csp_tpdo4(
            self.csp_options.feedback_timeout_ms / 1000.0
        )
        if status & 0x0008:
            raise DriveError(f"Drive fault during CSP: 0x{status:04X}")
        if FaulhaberDrive.state(status) != 0x0027:
            raise DriveError(f"Drive left Operation Enabled: 0x{status:04X}")
        return Feedback(
            statusword=status,
            raw_position=raw,
            width_mm=raw_to_width_mm(raw, self.limits),
            current_ma=current,
        )

    def cycle_width_mm(self, width_mm: float) -> Feedback:
        self.command_width_mm(width_mm)
        return self.read_feedback()

    def stop(self, *, quick: bool = False) -> None:
        """Best-effort stop and exact restoration of temporary drive settings."""
        drive = self.drive
        if drive is not None:
            if quick:
                try:
                    drive.quick_stop()
                    print("Quick Stop requested")
                except Exception:
                    pass
            try:
                drive.disable()
            except Exception:
                pass
            if self._saved_tpdo4 is not None:
                try:
                    drive.restore_tpdo4(self._saved_tpdo4)
                    print("Original TPDO4 mapping restored")
                except Exception as error:
                    print(f"WARNING: Could not restore TPDO4 mapping: {error}")
            if self._saved_rpdo1 is not None:
                try:
                    drive.restore_rpdo1(self._saved_rpdo1)
                    print("Original RPDO1 mapping restored")
                except Exception as error:
                    print(f"WARNING: Could not restore RPDO1 mapping: {error}")
            if self._original_mode is not None:
                try:
                    drive.write_u8(0x6060, self._original_mode & 0xFF)
                    print(f"Original mode {self._original_mode} restored")
                except Exception as error:
                    print(f"WARNING: Could not restore original mode: {error}")
            if self._original_continuous_current is not None:
                try:
                    drive.write_u16(0x2333, self._original_continuous_current, 1)
                    drive.write_u16(0x2333, int(self._original_peak_current), 2)
                    print("Original current limits restored")
                except Exception as error:
                    print(f"WARNING: Could not restore current limits: {error}")
        if self.bus is not None:
            try:
                self.bus.shutdown()
            except Exception:
                pass
        self._started = False
        self._saved_rpdo1 = None
        self._saved_tpdo4 = None
        self.drive = None
        self.bus = None
