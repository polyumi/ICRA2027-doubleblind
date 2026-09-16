#!/usr/bin/env python3
"""Track trajectory_msgs/JointTrajectory chunks with a FAULHABER gripper.

The ROS callback only validates and buffers trajectories.  A dedicated thread
runs the CANopen CSP loop at a fixed monotonic rate, evaluates the trajectory in
ROS time, adds an experimentally measured fixed-delay lead, and sends the
interpolated width target to the motor.

For rosbag playback, run this node with ``use_sim_time:=true`` and play the bag
with a high-rate /clock publisher (see the commands printed by --help).
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import math
from pathlib import Path
import signal
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from .faulhaber_csp_motor import (
    CSPOptions,
    CalibrationOptions,
    DriveError,
    FaulhaberCSPMotor,
)


@dataclass(frozen=True, order=True)
class TimedWidth:
    ros_time_ns: int
    width_mm: float
    is_source_waypoint: bool = True


class TrajectoryBuffer:
    """Thread-safe, replaceable future waypoint buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: list[TimedWidth] = []
        self._last_receive_mono: float | None = None

    def replace_from(
        self,
        points: list[TimedWidth],
        receive_ros_ns: int,
        current_width_mm: float,
    ) -> None:
        first = points[0].ros_time_ns
        with self._lock:
            # A newer receding-horizon chunk supersedes its overlapping future.
            kept = [point for point in self._points if point.ros_time_ns < first]
            # If the old buffer ended before this chunk arrived, hold the old final
            # command until receipt and ramp only across the available lookahead.
            # This avoids interpolating across a period when the new plan was not
            # yet known (notably the bag's initial one-point command).
            if (
                self._points
                and self._points[-1].ros_time_ns < receive_ros_ns < first
            ):
                kept.append(
                    TimedWidth(
                        receive_ros_ns,
                        self._points[-1].width_mm,
                        is_source_waypoint=False,
                    )
                )
            elif not self._points and receive_ros_ns < first:
                kept.append(
                    TimedWidth(
                        receive_ros_ns,
                        current_width_mm,
                        is_source_waypoint=False,
                    )
                )
            merged = kept + points
            # Collapse duplicate timestamps in favor of the newest chunk.
            by_time = {point.ros_time_ns: point for point in merged}
            self._points = [by_time[key] for key in sorted(by_time)]
            self._last_receive_mono = time.monotonic()

    def clear(self) -> None:
        with self._lock:
            self._points.clear()
            self._last_receive_mono = None

    def sample(self, query_ns: int) -> tuple[float | None, str]:
        with self._lock:
            points = tuple(self._points)
        if not points:
            return None, "empty"
        times = [point.ros_time_ns for point in points]
        index = bisect_right(times, query_ns)
        if index == 0:
            return points[0].width_mm, "before"
        if index == len(points):
            return points[-1].width_mm, "after"
        left = points[index - 1]
        right = points[index]
        span = right.ros_time_ns - left.ros_time_ns
        if span <= 0:
            return right.width_mm, "exact"
        fraction = (query_ns - left.ros_time_ns) / span
        return left.width_mm + fraction * (right.width_mm - left.width_mm), "interpolated"

    def last_receive_age(self) -> float | None:
        with self._lock:
            stamp = self._last_receive_mono
        return None if stamp is None else time.monotonic() - stamp

    def source_waypoints_between(
        self,
        start_exclusive_ns: int,
        end_inclusive_ns: int,
    ) -> list[TimedWidth]:
        """Return original trajectory waypoints crossed during a ROS-time interval."""
        if end_inclusive_ns <= start_exclusive_ns:
            return []
        with self._lock:
            return [
                point
                for point in self._points
                if point.is_source_waypoint
                and start_exclusive_ns < point.ros_time_ns <= end_inclusive_ns
            ]

    def bounds(self) -> tuple[int, int] | None:
        with self._lock:
            if not self._points:
                return None
            return self._points[0].ros_time_ns, self._points[-1].ros_time_ns


class FaulhaberTrajectoryNode(Node):
    def __init__(
        self,
        args: argparse.Namespace,
        motor: FaulhaberCSPMotor,
    ) -> None:
        super().__init__("faulhaber_gripper_trajectory")
        self.args = args
        self.motor = motor
        self.buffer = TrajectoryBuffer()
        self.shutdown_event = threading.Event()
        self.worker_stop_event = threading.Event()
        self.worker_error: BaseException | None = None
        self.worker: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_warning_mono = 0.0
        self._message_count = 0
        self._current_command_mm: float | None = None
        self._tracking_ready = False
        self.subscription = self.create_subscription(
            JointTrajectory,
            args.topic,
            self._trajectory_callback,
            10,
        )
        self.joint_state_publisher = self.create_publisher(
            JointState,
            args.joint_state_topic,
            10,
        )
        self.current_publisher = self.create_publisher(
            Float64,
            args.current_topic,
            10,
        )
        self.waypoint_error_publisher = self.create_publisher(
            Float64,
            args.waypoint_error_topic,
            10,
        )
        self.calibration_service = self.create_service(
            Trigger,
            args.calibration_service,
            self._calibration_callback,
        )

    def start_tracking(self, initial_width_mm: float) -> None:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("CSP scheduler is already running")
        with self._state_lock:
            self._current_command_mm = initial_width_mm
            self._tracking_ready = True
        self.worker_error = None
        self.worker_stop_event = threading.Event()
        self.worker = threading.Thread(
            target=self._control_loop,
            name="faulhaber-csp-loop",
            daemon=False,
        )
        self.worker.start()

    def stop_tracking(self) -> None:
        with self._state_lock:
            self._tracking_ready = False
        self.worker_stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=2.0)

    def request_stop(self) -> None:
        self.shutdown_event.set()
        self.stop_tracking()

    def current_command_mm(self) -> float | None:
        with self._state_lock:
            if not self._tracking_ready:
                return None
            return self._current_command_mm

    def _warn_throttled(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_mono >= 1.0:
            self.get_logger().warning(text)
            self._last_warning_mono = now

    def _publish_joint_state(self, width_mm: float) -> None:
        """Publish the measured gripper opening as a standard ROS joint state."""
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = [self.args.joint_name]
        message.position = [width_mm / 1000.0]
        # Velocity is not included in the synchronous feedback PDO, and motor
        # current is not joint effort, so both fields intentionally remain empty.
        message.velocity = []
        message.effort = []
        self.joint_state_publisher.publish(message)

    def _publish_current(self, current_ma: int) -> None:
        """Publish the most recent TPDO motor-current measurement."""
        message = Float64()
        message.data = float(current_ma)
        self.current_publisher.publish(message)

    def _publish_waypoint_error(
        self,
        waypoint: TimedWidth,
        actual_width_mm: float,
    ) -> None:
        """Publish tracking error only at an original waypoint timestamp."""
        error_message = Float64()
        error_message.data = waypoint.width_mm - actual_width_mm
        self.waypoint_error_publisher.publish(error_message)

    def _calibration_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request
        if not self._lifecycle_lock.acquire(blocking=False):
            response.success = False
            response.message = "Calibration or motor transition already in progress"
            return response
        try:
            self.get_logger().warning(
                "Calibration requested: stopping trajectory tracking and moving "
                "to both mechanical hard stops"
            )
            self.stop_tracking()
            self.buffer.clear()
            feedback = self.motor.calibrate_and_restart()
            self._message_count = 0
            self.start_tracking(feedback.width_mm)
            response.success = True
            response.message = (
                f"Calibration complete; holding {feedback.width_mm:.2f} mm and "
                "waiting for a new trajectory chunk"
            )
            self.get_logger().info(response.message)
        except BaseException as error:
            self.motor.stop(quick=True)
            response.success = False
            response.message = f"Calibration failed: {error}"
            self.get_logger().error(response.message)
        finally:
            self._lifecycle_lock.release()
        return response

    def _trajectory_callback(self, message: JointTrajectory) -> None:
        current_width = self.current_command_mm()
        if current_width is None:
            self._warn_throttled(
                "Ignoring trajectory because the motor is not calibrated and ready"
            )
            return
        if self.args.joint_name not in message.joint_names:
            self._warn_throttled(
                f"Ignoring trajectory without joint {self.args.joint_name!r}: "
                f"{list(message.joint_names)}"
            )
            return
        if not message.points:
            self._warn_throttled("Ignoring empty JointTrajectory")
            return
        joint_index = message.joint_names.index(self.args.joint_name)
        header_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        # A zero header stamp means "start now" in common ROS trajectory usage.
        if header_ns == 0:
            header_ns = self.get_clock().now().nanoseconds

        parsed: list[TimedWidth] = []
        previous_ns: int | None = None
        previous_width: float | None = None
        for index, point in enumerate(message.points):
            if joint_index >= len(point.positions):
                self._warn_throttled(
                    f"Ignoring malformed point {index}: positions has length "
                    f"{len(point.positions)}"
                )
                return
            offset_ns = (
                int(point.time_from_start.sec) * 1_000_000_000
                + int(point.time_from_start.nanosec)
            )
            absolute_ns = header_ns + offset_ns
            width_m = float(point.positions[joint_index])
            if not math.isfinite(width_m):
                self._warn_throttled(f"Ignoring non-finite width at point {index}")
                return
            width_mm = width_m * 1000.0
            if width_mm < 0.0 or width_mm > self.args.max_width_mm:
                self._warn_throttled(
                    f"Clamping width {width_mm:.3f} mm into "
                    f"[0, {self.args.max_width_mm:g}] mm"
                )
                width_mm = min(self.args.max_width_mm, max(0.0, width_mm))
            if previous_ns is not None:
                if absolute_ns <= previous_ns:
                    self._warn_throttled("Ignoring trajectory with non-increasing times")
                    return
                dt = (absolute_ns - previous_ns) * 1e-9
                speed_mm_s = abs(width_mm - float(previous_width)) / dt
                rpm = (
                    speed_mm_s
                    * self.args.raw_span_estimate
                    / self.args.max_width_mm
                    * 60.0
                    / self.args.increments_per_revolution
                )
                if rpm > self.args.max_command_speed_rpm:
                    self._warn_throttled(
                        f"Rejecting trajectory segment requesting approximately "
                        f"{rpm:.1f} rpm; limit={self.args.max_command_speed_rpm:g} rpm"
                    )
                    return
            parsed.append(TimedWidth(absolute_ns, width_mm))
            previous_ns = absolute_ns
            previous_width = width_mm

        receive_ns = self.get_clock().now().nanoseconds
        if self._message_count == 0:
            # Start this playback from the physical gripper state, rather than
            # jumping to a recorded single-point initial condition.
            parsed[0] = TimedWidth(parsed[0].ros_time_ns, current_width)
        self.buffer.replace_from(parsed, receive_ns, current_width)
        self._message_count += 1
        if self._message_count == 1:
            lead_ms = (parsed[0].ros_time_ns - self.get_clock().now().nanoseconds) / 1e6
            self.get_logger().info(
                f"First trajectory: {len(parsed)} point(s), first-point "
                f"lookahead={lead_ms:.1f} ms"
            )

    def _control_loop(self) -> None:
        period = 1.0 / self.args.csp_rate_hz
        delay_ns = int(
            round(self.args.delay_compensation_ms * 1_000_000.0)
        )

        next_cycle = time.monotonic()
        next_telemetry = next_cycle
        last_waypoint_evaluation_ns = self.get_clock().now().nanoseconds

        held_width = self.current_command_mm()
        if held_width is None:
            self.worker_error = DriveError(
                "CSP scheduler started without a position"
            )
            self.shutdown_event.set()
            return

        try:
            while not self.worker_stop_event.is_set():
                next_cycle += period
                remaining = next_cycle - time.monotonic()

                if remaining > 0:
                    self.worker_stop_event.wait(remaining)

                    if self.worker_stop_event.is_set():
                        break
                else:
                    lateness_ms = -remaining * 1000.0

                    if lateness_ms > self.args.max_cycle_lateness_ms:
                        raise DriveError(
                            f"CSP scheduler late by {lateness_ms:.1f} ms; "
                            f"limit={self.args.max_cycle_lateness_ms:g} ms"
                        )

                query_ns = (
                    self.get_clock().now().nanoseconds
                    + delay_ns
                )

                desired_mm, state = self.buffer.sample(query_ns)

                if desired_mm is None:
                    # No trajectory is buffered yet. Hold the position at
                    # which CSP tracking was started.
                    desired_mm = held_width
                else:
                    held_width = desired_mm

                with self._state_lock:
                    self._current_command_mm = held_width

                feedback = self.motor.cycle_width_mm(desired_mm)
                self._publish_joint_state(feedback.width_mm)
                self._publish_current(feedback.current_ma)
                feedback_ros_ns = self.get_clock().now().nanoseconds
                if feedback_ros_ns < last_waypoint_evaluation_ns:
                    # Handles a rosbag loop, seek, or simulated-clock reset without
                    # replaying every waypoint from the previous time epoch.
                    last_waypoint_evaluation_ns = feedback_ros_ns
                else:
                    for waypoint in self.buffer.source_waypoints_between(
                        last_waypoint_evaluation_ns,
                        feedback_ros_ns,
                    ):
                        self._publish_waypoint_error(
                            waypoint,
                            feedback.width_mm,
                        )
                    last_waypoint_evaluation_ns = feedback_ros_ns
                error_mm = desired_mm - feedback.width_mm

                # A value of zero disables following-error shutdown. This
                # allows an object to prevent the gripper from reaching its
                # requested width while the controller limits gripping current.
                if (
                    self.args.max_tracking_error_mm > 0.0
                    and abs(error_mm)
                    > self.args.max_tracking_error_mm
                ):
                    raise DriveError(
                        f"Following error {error_mm:+.2f} mm exceeds "
                        f"{self.args.max_tracking_error_mm:g} mm"
                    )

                if time.monotonic() >= next_telemetry:
                    age = self.buffer.last_receive_age()
                    age_text = (
                        "none"
                        if age is None
                        else f"{age:.3f}s"
                    )

                    tracking_limit_text = (
                        "disabled"
                        if self.args.max_tracking_error_mm == 0
                        else f"{self.args.max_tracking_error_mm:g}mm"
                    )

                    self.get_logger().debug(
                        f"desired={desired_mm:7.2f} mm "
                        f"actual={feedback.width_mm:7.2f} mm "
                        f"error={error_mm:+6.2f} mm "
                        f"current={feedback.current_ma:4d} mA "
                        f"buffer={state} "
                        f"message_age={age_text} "
                        f"error_limit={tracking_limit_text}"
                    )

                    next_telemetry = (
                        time.monotonic()
                        + 1.0 / self.args.telemetry_hz
                    )

        except BaseException as error:
            self.worker_error = error
            self.shutdown_event.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track ROS JointTrajectory chunks with a FAULHABER CSP gripper",
        epilog=(
            "Playback example:\n"
            "  ros2 bag play BAG_DIRECTORY --clock 200\n"
            "The node must use simulated time: --ros-args -p use_sim_time:=true"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topic", default="/polyumi/target_gripper")
    parser.add_argument("--joint-name", default="fr3_gripper_width")
    parser.add_argument(
        "--joint-state-topic",
        default="/faulhaber_gripper/joint_states",
    )
    parser.add_argument(
        "--current-topic",
        default="/faulhaber_gripper/motor_current_ma",
    )
    parser.add_argument(
        "--waypoint-error-topic",
        default="/faulhaber_gripper/waypoint_tracking_error_mm",
    )
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--node-id", type=int, default=1)
    parser.add_argument("--calibration-file", type=Path, default=Path("faulhaber_gripper_limits.json"))
    parser.add_argument(
        "--calibration-service",
        default="/faulhaber_gripper/calibrate",
    )
    parser.add_argument("--max-width-mm", type=float, default=81.2)
    parser.add_argument("--current-limit-ma", type=int, default=300)
    parser.add_argument("--stall-current-ma", type=int, default=220)
    parser.add_argument("--first-speed-rpm", type=int, default=60)
    parser.add_argument("--second-speed-rpm", type=int, default=10)
    parser.add_argument("--backoff-increments", type=int, default=100)
    parser.add_argument("--stall-duration", type=float, default=0.4)
    parser.add_argument("--max-open-travel", type=int, default=1_000_000)
    parser.add_argument("--max-close-travel", type=int, default=2_000_000)
    parser.add_argument("--homing-timeout", type=float, default=180.0)
    parser.add_argument("--endpoint-margin", type=int, default=100)
    parser.add_argument("--csp-current-limit-ma", type=int, default=400)
    parser.add_argument("--csp-rate-hz", type=float, default=200.0)
    parser.add_argument("--feedback-timeout-ms", type=float, default=4.0)
    parser.add_argument("--delay-compensation-ms", type=float, default=13.3)
    parser.add_argument("--telemetry-hz", type=float, default=5.0)
    parser.add_argument("--max-tracking-error-mm", type=float, default=0.0)
    parser.add_argument("--max-cycle-lateness-ms", type=float, default=20.0)
    parser.add_argument("--max-command-speed-rpm", type=float, default=4000.0)
    parser.add_argument("--increments-per-revolution", type=float, default=3000.0)
    parser.add_argument(
        "--raw-span-estimate",
        type=float,
        default=121000.0,
        help="Conservative raw stroke estimate used only for preflight speed checks",
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not 1 <= args.node_id <= 127:
        parser.error("--node-id must be from 1 through 127")

    positive = (
        "max_width_mm",
        "current_limit_ma",
        "stall_current_ma",
        "first_speed_rpm",
        "second_speed_rpm",
        "max_open_travel",
        "max_close_travel",
        "homing_timeout",
        "csp_current_limit_ma",
        "csp_rate_hz",
        "feedback_timeout_ms",
        "telemetry_hz",
        "max_cycle_lateness_ms",
        "max_command_speed_rpm",
        "increments_per_revolution",
        "raw_span_estimate",
    )

    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be positive"
            )

    # Zero disables software following-error shutdown.
    if args.max_tracking_error_mm < 0:
        parser.error(
            "--max-tracking-error-mm must be zero or positive"
        )

    if args.stall_current_ma >= args.current_limit_ma:
        parser.error(
            "--stall-current-ma must be below --current-limit-ma"
        )

    if args.delay_compensation_ms < 0:
        parser.error(
            "--delay-compensation-ms must not be negative"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ros_args = rclpy.utilities.remove_ros_args(args=sys.argv if argv is None else argv)
    args = parser.parse_args(ros_args[1:])
    validate_args(parser, args)

    calibration = CalibrationOptions(
        current_limit_ma=args.current_limit_ma,
        stall_current_ma=args.stall_current_ma,
        first_speed_rpm=args.first_speed_rpm,
        second_speed_rpm=args.second_speed_rpm,
        backoff_increments=args.backoff_increments,
        stall_duration=args.stall_duration,
        max_open_travel=args.max_open_travel,
        max_close_travel=args.max_close_travel,
        homing_timeout=args.homing_timeout,
        endpoint_margin=args.endpoint_margin,
        max_width_mm=args.max_width_mm,
    )
    motor = FaulhaberCSPMotor(
        interface=args.interface,
        node_id=args.node_id,
        calibration=calibration,
        csp=CSPOptions(
            current_limit_ma=args.csp_current_limit_ma,
            feedback_timeout_ms=args.feedback_timeout_ms,
        ),
        calibration_path=args.calibration_file,
        recalibrate=False,
    )

    rclpy.init(args=sys.argv if argv is None else argv)
    node = FaulhaberTrajectoryNode(args, motor)
    try:
        try:
            feedback = motor.start()
            node.start_tracking(feedback.width_mm)
            print(
                f"CSP ready at the current position ({feedback.width_mm:.2f} mm). "
                "Start bag playback now."
            )
        except Exception as error:
            node.get_logger().warning(
                f"Motor tracking is not ready: {error}. Clear the mechanism, then "
                f"call {args.calibration_service} to calibrate and start CSP."
            )

        while rclpy.ok() and not node.shutdown_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.worker_error is not None:
            raise node.worker_error
        return 0
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    except (DriveError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        node.request_stop()
        node.destroy_node()
        motor.stop(quick=bool(node.worker_error))
        if rclpy.ok():
            rclpy.shutdown()
        print("Drive disabled; exiting")


if __name__ == "__main__":
    raise SystemExit(main())
