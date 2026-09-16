#!/usr/bin/env python3
"""Read-only health check for a FAULHABER CANopen motor controller.

The script never changes NMT state, CiA 402 state, operating mode, or target.
It sends only node-guard remote requests and SDO upload (read) requests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import subprocess
import time

import can


@dataclass
class LinkState:
    text: str
    can_state: str | None
    tx_error: int | None
    rx_error: int | None
    bus_errors: int | None
    bus_off: int | None


class CheckError(RuntimeError):
    pass


def read_link_state(interface: str) -> LinkState:
    result = subprocess.run(
        ["ip", "-details", "-statistics", "link", "show", interface],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CheckError(result.stderr.strip() or f"Cannot inspect {interface}")

    text = result.stdout
    state_match = re.search(r"can state ([A-Z-]+)", text)
    counter_match = re.search(r"berr-counter tx (\d+) rx (\d+)", text)
    totals_match = re.search(
        r"re-started\s+bus-errors\s+arbit-lost\s+error-warn\s+"
        r"error-pass\s+bus-off\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+"
        r"(\d+)\s+(\d+)\s+(\d+)",
        text,
    )
    return LinkState(
        text=text,
        can_state=state_match.group(1) if state_match else None,
        tx_error=int(counter_match.group(1)) if counter_match else None,
        rx_error=int(counter_match.group(2)) if counter_match else None,
        bus_errors=int(totals_match.group(2)) if totals_match else None,
        bus_off=int(totals_match.group(6)) if totals_match else None,
    )


class ReadOnlyCanopen:
    def __init__(self, bus: can.BusABC, node_id: int, timeout: float):
        self.bus = bus
        self.node_id = node_id
        self.timeout = timeout
        self.sdo_request_id = 0x600 + node_id
        self.sdo_response_id = 0x580 + node_id
        self.heartbeat_id = 0x700 + node_id
        self.emergency_id = 0x080 + node_id
        self.boot_count = 0
        self.emergencies: list[tuple[int, bytes]] = []
        self.error_frames = 0

    def _send(self, arbitration_id: int, data: bytes = b"", *, remote: bool = False) -> None:
        self.bus.send(
            can.Message(
                arbitration_id=arbitration_id,
                data=data,
                is_extended_id=False,
                is_remote_frame=remote,
            )
        )

    def _record_async(self, message: can.Message) -> None:
        if message.is_error_frame:
            self.error_frames += 1
            return
        data = bytes(message.data)
        if message.arbitration_id == self.heartbeat_id and data == b"\x00":
            self.boot_count += 1
        elif message.arbitration_id == self.emergency_id and len(data) == 8:
            code = int.from_bytes(data[0:2], "little")
            self.emergencies.append((code, data))

    def _receive_until(self, predicate, deadline: float) -> can.Message:
        while time.monotonic() < deadline:
            message = self.bus.recv(max(0.0, deadline - time.monotonic()))
            if message is None:
                break
            self._record_async(message)
            if predicate(message):
                return message
        raise CheckError("Timed out waiting for CANopen response")

    def node_guard(self) -> tuple[int, float]:
        started = time.monotonic()
        self._send(self.heartbeat_id, remote=True)
        message = self._receive_until(
            lambda msg: (
                not msg.is_error_frame
                and msg.arbitration_id == self.heartbeat_id
                and not msg.is_remote_frame
                and len(msg.data) == 1
            ),
            started + self.timeout,
        )
        state = message.data[0] & 0x7F
        return state, (time.monotonic() - started) * 1000.0

    def sdo_read(
        self, index: int, subindex: int = 0, *, signed: bool = False
    ) -> tuple[int, float]:
        request = bytes((0x40, index & 0xFF, index >> 8, subindex, 0, 0, 0, 0))
        started = time.monotonic()
        self._send(self.sdo_request_id, request)

        def matches(message: can.Message) -> bool:
            data = bytes(message.data)
            return (
                not message.is_error_frame
                and message.arbitration_id == self.sdo_response_id
                and len(data) == 8
                and data[1] == (index & 0xFF)
                and data[2] == (index >> 8)
                and data[3] == subindex
            )

        message = self._receive_until(matches, started + self.timeout)
        data = bytes(message.data)
        if data[0] == 0x80:
            abort = int.from_bytes(data[4:8], "little")
            raise CheckError(
                f"SDO abort 0x{abort:08X} reading 0x{index:04X}:{subindex:02X}"
            )
        sizes = {0x4F: 1, 0x4B: 2, 0x47: 3, 0x43: 4}
        size = sizes.get(data[0])
        if size is None:
            raise CheckError(f"Unexpected SDO response command 0x{data[0]:02X}")
        value = int.from_bytes(data[4 : 4 + size], "little", signed=signed)
        return value, (time.monotonic() - started) * 1000.0

    def drain(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            message = self.bus.recv(min(0.05, deadline - time.monotonic()))
            if message is not None:
                self._record_async(message)


NMT_NAMES = {
    0x00: "Boot-up",
    0x04: "Stopped",
    0x05: "Operational",
    0x7F: "Pre-operational",
}


def cia402_state(statusword: int) -> str:
    masked = statusword & 0x006F
    return {
        0x0000: "Not ready to switch on",
        0x0040: "Switch on disabled",
        0x0021: "Ready to switch on",
        0x0023: "Switched on",
        0x0027: "Operation enabled",
        0x0007: "Quick stop active",
        0x000F: "Fault reaction active",
        0x0008: "Fault",
    }.get(masked, f"Unknown state mask 0x{masked:04X}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only FAULHABER CANopen health check")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--node-id", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--response-timeout", type=float, default=1.0)
    args = parser.parse_args()

    if not 1 <= args.node_id <= 127:
        parser.error("node ID must be from 1 through 127")
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    failures: list[str] = []
    latencies: list[float] = []

    try:
        before = read_link_state(args.interface)
    except CheckError as error:
        print(f"FAIL: {error}")
        return 1

    print(
        f"Interface {args.interface}: state={before.can_state}, "
        f"tx_err={before.tx_error}, rx_err={before.rx_error}"
    )
    if before.can_state != "ERROR-ACTIVE":
        failures.append(f"CAN interface began in {before.can_state}")
    if (before.tx_error or 0) != 0 or (before.rx_error or 0) != 0:
        failures.append("CAN interface began with nonzero live error counters")

    try:
        with can.Bus(
            interface="socketcan",
            channel=args.interface,
            receive_own_messages=False,
        ) as bus:
            drive = ReadOnlyCanopen(bus, args.node_id, args.response_timeout)

            try:
                nmt_state, latency = drive.node_guard()
                latencies.append(latency)
                print(
                    f"Node {args.node_id}: {NMT_NAMES.get(nmt_state, f'0x{nmt_state:02X}')} "
                    f"({latency:.2f} ms)"
                )
            except CheckError as error:
                failures.append(f"Node guarding: {error}")

            reads = (
                ("Device type", 0x1000, 0, False),
                ("Statusword", 0x6041, 0, False),
                ("Active mode", 0x6061, 0, True),
                ("Position", 0x6064, 0, True),
                ("Velocity", 0x606C, 0, True),
                ("Motor current", 0x6078, 0, True),
            )
            values: dict[str, int] = {}
            for label, index, subindex, signed in reads:
                try:
                    value, latency = drive.sdo_read(index, subindex, signed=signed)
                    values[label] = value
                    latencies.append(latency)
                    suffix = " mA" if label == "Motor current" else ""
                    print(f"{label}: {value}{suffix} ({latency:.2f} ms)")
                except CheckError as error:
                    failures.append(f"{label}: {error}")

            if "Statusword" in values:
                status = values["Statusword"]
                print(f"CiA 402: {cia402_state(status)} (0x{status:04X})")
                if status & 0x0008:
                    failures.append(f"Drive statusword reports a fault: 0x{status:04X}")

            print(f"Monitoring {args.samples} read-only samples...")
            for sample in range(1, args.samples + 1):
                try:
                    status, latency = drive.sdo_read(0x6041)
                    latencies.append(latency)
                    if status & 0x0008:
                        failures.append(f"Sample {sample}: drive fault 0x{status:04X}")
                except CheckError as error:
                    failures.append(f"Sample {sample}: {error}")
                drive.drain(args.interval)

            if drive.boot_count:
                failures.append(f"Observed {drive.boot_count} controller boot/reset message(s)")
            nonzero_emergency = [item for item in drive.emergencies if item[0] != 0]
            if nonzero_emergency:
                codes = ", ".join(f"0x{code:04X}" for code, _ in nonzero_emergency)
                failures.append(f"Observed nonzero emergency code(s): {codes}")
            if drive.error_frames:
                failures.append(f"Observed {drive.error_frames} CAN error frame(s)")

    except (can.CanError, OSError) as error:
        failures.append(f"SocketCAN: {error}")

    try:
        after = read_link_state(args.interface)
        print(
            f"Final interface: state={after.can_state}, "
            f"tx_err={after.tx_error}, rx_err={after.rx_error}"
        )
        if after.can_state != "ERROR-ACTIVE":
            failures.append(f"CAN interface ended in {after.can_state}")
        if (after.tx_error or 0) != 0 or (after.rx_error or 0) != 0:
            failures.append("CAN interface ended with nonzero live error counters")
        if (
            before.bus_errors is not None
            and after.bus_errors is not None
            and after.bus_errors > before.bus_errors
        ):
            failures.append("Kernel bus-error counter increased during the test")
        if (
            before.bus_off is not None
            and after.bus_off is not None
            and after.bus_off > before.bus_off
        ):
            failures.append("Kernel bus-off counter increased during the test")
    except CheckError as error:
        failures.append(f"Final interface inspection: {error}")

    if latencies:
        print(
            f"Response latency: min={min(latencies):.2f} ms, "
            f"max={max(latencies):.2f} ms, "
            f"mean={sum(latencies) / len(latencies):.2f} ms"
        )

    if failures:
        print("\nRESULT: UNHEALTHY")
        for failure in failures:
            print(f"  - {failure}")
        print("No enable, mode-write, target, or movement command was sent.")
        return 1

    print("\nRESULT: HEALTHY")
    print("Node guarding, SDO reads, controller state, and CAN counters passed.")
    print("No enable, mode-write, target, or movement command was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
