#!/usr/bin/env python3
"""
Water-bottle shake: level the tool, then shake it vertically a fixed number of times.

Built to produce a labelled audio corpus. The arm holds a bottle, shakes it the SAME way every
run, and a bag records the contact mic, the finger camera and the GoPro while it does. The water
level is the label -- it comes from how the bottle was filled, not from anything measured here --
so the only thing this node owes the dataset is that every trial move identically. Everything
below is in service of that: a fixed waypoint grid, a closed-form height profile, and a starting
orientation derived from the robot rather than from wherever the operator left it.

**Levelling.** ``polyumi_tcp`` is in GoPro-optical axes (x right, y down, z forward, and z is the
approach axis -- see nuc/tcp_calib.py). "Perfectly horizontal" here means:

    z' (approach)     horizontal, pointing the way it already pointed
    y' (optical down) straight down, along world -Z
    x' = y' x z'      horizontal

i.e. the tool's yaw is kept and its roll and pitch are zeroed. The bottle ends up held level,
facing the same direction as before, which is the least surprising thing to do to a pose the
operator chose. The current approach axis is projected onto the world XY plane to get that yaw,
so a tool already pointing straight up or down has no defined yaw and the node refuses rather
than picking one.

**The shake** is a raised cosine in world Z about the levelled start:

    z(t) = z0 + A * (1 - cos(2*pi*t/T)) / 2

One period is one shake: up to +A at T/2 and back to z0 at T. It never goes BELOW z0, so the
motion cannot drive the bottle into the table even if the start pose is low. Position x and y and
the orientation are held fixed throughout -- only height changes.

Usage (laptop, after `source setup_franka_env.sh`):

    # 1. NUC: bringup + inference, arm execution on.
    ros2 launch nuc/launch/fr3_inference.launch.py execute_arm:=true

    # 2. Put the arm somewhere roomy with the bottle gripped, then DRY RUN (default): nothing
    #    moves, and the whole commanded path shows up on /polyumi/target_poses_preview.
    ros2 run polyumi_ros2 water_shake

    # 3. Watch the preview in Foxglove. When it looks right, execute:
    ros2 run polyumi_ros2 water_shake --ros-args -p execute:=true

    # 4. Record a trial (separate shell), then run step 3 again:
    ros2_ws/src/polyumi_ros2/scripts/record_eval_trial.sh water_half_full

**Do not run this while policy_client_node is running.** Both publish to
/polyumi/target_poses_traj and the controller splices whichever chunk arrives last.
"""

import math
import sys
import threading
import time

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from polyumi_ros2.gripper_map import aperture_from_joint_state
from polyumi_ros2.target_chunk import CONSUMER_HINT, TargetChunkPublisher
from tf2_ros import Buffer, TransformListener

#: Below this, the approach axis is too close to vertical for its projection onto the world XY
#: plane to define a yaw. 0.1 rad of tilt off vertical still leaves a 0.0998 horizontal component,
#: so this only rejects a tool genuinely pointing up or down.
MIN_HORIZONTAL_APPROACH = 0.1

PREVIEW_TOPIC = '/polyumi/target_poses_preview'

#: What the streaming impedance controller will actually allow, from nuc/config/polyumi_controllers.yaml:
#: translational_clip 0.01 m x translational_stiffness 2000 N/m = 20 N, and max_pos_speed 1.0 m/s.
#: With the PAYLOAD_MASS in nuc/tcp_calib.py that is ~29 m/s^2 -- about 2.9 g, which is plenty for
#: sloshing. The binding constraint on a vigorous shake is the speed cap, not the force.
MAX_FORCE_N = 20.0
MAX_SPEED_MPS = 1.0
PAYLOAD_KG = 0.7
GRAVITY = 9.81
GRIPPER_TOPIC = '/polyumi/target_gripper'
GRIPPER_STATE_TOPIC = '/fr3_gripper/joint_states'
GRIPPER_JOINT_NAME = 'fr3_gripper_width'


def level_rotation(approach_world: np.ndarray) -> Rotation:
    """
    Build the levelled TCP orientation that keeps `approach_world`'s yaw and zeroes roll/pitch.

    :param approach_world: the TCP's current z axis (approach) in the base frame, any length.
    :returns: the levelled rotation, as base_R_tcp.
    :raises ValueError: if the approach axis is too near vertical to define a yaw.

    The returned frame is right-handed by construction: x' = y' x z' with y' straight down.
    """
    approach = np.asarray(approach_world, dtype=float)
    horizontal = float(np.hypot(approach[0], approach[1]))
    if horizontal < MIN_HORIZONTAL_APPROACH:
        raise ValueError(
            f'approach axis is {horizontal:.3f} from vertical in the XY plane (limit '
            f'{MIN_HORIZONTAL_APPROACH}); it points up or down, so "keep the yaw" means nothing. '
            'Rotate the tool to point roughly sideways first.'
        )
    yaw = math.atan2(approach[1], approach[0])
    z_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    return Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))


def shake_heights(amplitude_m: float, n_shakes: int, period_s: float, dt: float, symmetric: bool = False) -> np.ndarray:
    """
    Displacements along the shake axis, one per waypoint, for `n_shakes` cycles.

    :returns: offsets in metres, starting and ending at 0.0.

    Two forms. The one-sided raised cosine ``A(1-cos)/2`` never goes negative, which is what a
    VERTICAL shake needs so it cannot drive down into the table. The symmetric form ``A sin``
    centres the travel on the start, giving twice the peak-to-peak excursion for the same
    amplitude and the same peak speed -- the better shake along a horizontal axis, where there is
    nothing underneath to hit.

    The grid is closed-form rather than accumulated so every trial lands on the same instants:
    the classifier's whole premise is that the only thing differing between runs is the water.
    """
    if amplitude_m <= 0 or n_shakes < 1 or period_s <= 0 or dt <= 0:
        raise ValueError('amplitude_m, period_s and dt must be > 0 and n_shakes >= 1')
    total_s = n_shakes * period_s
    n_steps = int(round(total_s / dt))
    t = np.arange(n_steps + 1) * dt
    if symmetric:
        return amplitude_m * np.sin(2.0 * math.pi * t / period_s)
    return amplitude_m * (1.0 - np.cos(2.0 * math.pi * t / period_s)) / 2.0


def slerp_quats(q_from: np.ndarray, q_to: np.ndarray, n: int) -> np.ndarray:
    """Interpolate `n` quaternions from `q_from` to `q_to` inclusive, as xyzw rows."""
    if n < 2:
        return np.asarray([q_to], dtype=float)
    key = Rotation.from_quat(np.vstack([q_from, q_to]))
    from scipy.spatial.transform import Slerp

    return Slerp([0.0, 1.0], key)(np.linspace(0.0, 1.0, n)).as_quat()


def _pose(position: np.ndarray, quat: np.ndarray) -> Pose:
    """Build a geometry_msgs Pose from a 3-vector and an xyzw quaternion."""
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in quat)
    return p


class WaterShakeNode(Node):
    """Level the TCP, then shake it vertically N times, as one absolutely-timed chunk."""

    def __init__(self, **kwargs):
        """Declare parameters and create the chunk and preview publishers."""
        super().__init__('water_shake', **kwargs)
        self.declare_parameter('amplitude_m', 0.08)
        self.declare_parameter('period_s', 1.0)
        self.declare_parameter('n_shakes', 5)
        self.declare_parameter('waypoint_dt', 0.05)
        self.declare_parameter('level_time_s', 2.0)
        self.declare_parameter('settle_s', 0.5)
        # Squeeze: command the jaws this much NARROWER than where they rest on the object, so the
        # driver's Move stalls against it and that stall becomes grip force. An object held at
        # exactly its measured width slips the moment the shake accelerates it. 2 mm is small on
        # purpose -- the payload is 0.7 kg and the aim is to stop slip, not crush a plastic bottle.
        #
        # This measure-then-squeeze path is CALIBRATION, to be run once per object. It is not safe
        # to repeat every trial: the jaws stay closed between trials, so the second trial measures
        # the width the first one squeezed to and takes another 2 mm off it. Across a session that
        # ratchets (76.9 mm to 73.7 mm over eight trials, observed), which is a monotonic change in
        # the setup that lands confounded with the class label whenever classes are collected in
        # blocks -- it made the finger camera encode trial number rather than the object.
        self.declare_parameter('grip_squeeze_m', 0.002)
        # The width to hold, in metres, once calibration has produced one. Set this for every trial
        # after the first so each grip is identical; 0 means "calibrate instead", i.e. measure what
        # the jaws rest at and take grip_squeeze_m off it. Either way the width actually commanded
        # is logged as GRIP_TARGET_M=<metres> so a collection script can capture it and hand it
        # back on the trials that follow.
        self.declare_parameter('grip_width_m', 0.0)
        # Where the shake starts from, in the base frame, and how far each trial is allowed to
        # wander from it. An unset home_xyz (all zeros) means "use wherever the arm is now", and that
        # used is logged as HOME_XYZ=x,y,z so a collection script can capture it once and hand it
        # back on every later trial.
        #
        # That hand-back is not optional. Jittering around the CURRENT pose instead of a recorded
        # one makes each trial start from the last trial's jittered position, so the start point
        # random-walks across the session -- a drift indistinguishable from the class label when
        # classes are collected in blocks, which is the failure this jitter exists to prevent.
        #
        # The point of the jitter: with a fixed start pose every trial photographs the same scene,
        # so a per-session offset in the camera's level or colour balance is the only thing that
        # varies and a classifier reads it instead of the object. Moving the start makes the view
        # vary far more than any such offset. Note this MASKS that nuisance rather than removing
        # it -- the offset belongs to the session, not the pose -- so it complements interleaving
        # the classes, it does not replace it.
        # All-zeros means "not set". It has to be a real triple of doubles rather than an empty
        # list, because rclpy infers a parameter's type from its default and an empty list infers
        # BYTE_ARRAY, which then rejects the coordinates this is meant to carry -- a failure that
        # appears only on the robot, the first time someone passes a real position. Zeros are an
        # unambiguous sentinel here: the base frame's origin is inside the robot, so the TCP can
        # never legitimately be there.
        self.declare_parameter('home_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('jitter_xyz_m', [0.0, 0.0, 0.0])
        self.declare_parameter('jitter_seed', -1)
        self.declare_parameter('grip', True)
        self.declare_parameter('grip_time_s', 1.0)
        # Direction to shake along, in the BASE frame. [0,0,1] is the original vertical shake.
        #
        # Sideways slosh far better, and the reason is not subtle: a vertical acceleration only
        # modulates effective gravity (g +/- a) and the surface stays level, while a HORIZONTAL
        # acceleration tilts effective gravity by atan(a/g) -- which is the thing that actually
        # moves water. At 11 m/s^2 that is a 48 degree tilt of the effective vertical.
        self.declare_parameter('shake_axis', [0.0, 0.0, 1.0])
        # Symmetric puts the start at the CENTRE of the travel rather than at one end, doubling
        # the peak-to-peak excursion for the same amplitude. Off by default because the original
        # vertical shake used the one-sided form to avoid driving down into the table; for a
        # horizontal axis there is no table and symmetric is the better shake.
        self.declare_parameter('symmetric', False)
        # Default false, like policy_client_node's execute_motion: running this by accident must
        # not move an arm holding a full bottle. The preview is published either way.
        self.declare_parameter('execute', False)
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'polyumi_tcp')

        self._p = {
            k: self.get_parameter(k).value
            for k in (
                'amplitude_m',
                'period_s',
                'n_shakes',
                'waypoint_dt',
                'level_time_s',
                'settle_s',
                'execute',
                'base_frame',
                'eef_frame',
                'grip_squeeze_m',
                'grip_width_m',
                'home_xyz',
                'jitter_xyz_m',
                'jitter_seed',
                'grip',
                'grip_time_s',
                'shake_axis',
                'symmetric',
            )
        }

        axis = np.asarray(self._p['shake_axis'], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError(f'shake_axis must be a finite non-zero 3-vector, got {axis}')
        self._axis = axis / np.linalg.norm(axis)

        self._grip_lock = threading.Lock()
        self._grip_width: float | None = None
        self.create_subscription(JointState, GRIPPER_STATE_TOPIC, self._on_gripper_state, 10)
        self._grip_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._preview = self.create_publisher(PoseArray, PREVIEW_TOPIC, 10)
        self._pub = (
            TargetChunkPublisher(self, frame_id=self._p['base_frame'], joint_name=self._p['eef_frame'])
            if self._p['execute']
            else None
        )

    def _on_gripper_state(self, msg: JointState) -> None:
        """Cache the jaw aperture reported by whichever gripper driver is running."""
        width = aperture_from_joint_state(msg)
        if width is not None:
            with self._grip_lock:
                self._grip_width = float(width)

    def _command_width(self, target: float) -> float:
        """Publish one absolute jaw width and report it in a form a script can capture."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [GRIPPER_JOINT_NAME]
        point = JointTrajectoryPoint()
        point.positions = [target]
        # The bridge sizes its move speed from this; ~1 s of allotted travel squeezes gently
        # rather than at the max speed a zero would ask for.
        point.time_from_start = Duration(sec=int(self._p['grip_time_s']), nanosec=0)
        msg.points.append(point)
        self._grip_pub.publish(msg)
        self.get_logger().info(f'GRIP_TARGET_M={target:.6f}')
        return target

    def squeeze(self, timeout_s: float = 10.0) -> float | None:
        """
        Hold `grip_width_m` if one is configured, otherwise calibrate one and return it.

        Calibration measures where the jaws rest on the object and takes `grip_squeeze_m` off
        that, because the right width depends on which object is in the gripper today. Run it
        once per object and pass the width it reports back as `grip_width_m` for every trial
        after, so every trial grips identically instead of tightening on the last one.

        Returns None only if calibration was needed and no gripper state arrived -- not fatal, so
        the caller warns and carries on rather than aborting a trial mid-collection.
        """
        fixed = float(self._p['grip_width_m'])
        if fixed > 0.0:
            self.get_logger().info(f'gripper: holding the configured {fixed * 1000:.1f}mm (no re-measure)')
            return self._command_width(fixed)

        deadline = time.monotonic() + timeout_s
        width = None
        while time.monotonic() < deadline:
            with self._grip_lock:
                width = self._grip_width
            if width is not None:
                break
            time.sleep(0.1)
        if width is None:
            self.get_logger().warn(
                f'no {GRIPPER_STATE_TOPIC} within {timeout_s:.0f}s -- not squeezing. The object '
                'may slip once the shake accelerates it.'
            )
            return None

        target = max(0.0, width - float(self._p['grip_squeeze_m']))
        self.get_logger().info(
            f'gripper: calibrating -- resting at {width * 1000:.1f}mm, commanding '
            f'{target * 1000:.1f}mm ({float(self._p["grip_squeeze_m"]) * 1000:.1f}mm squeeze). '
            f'Pass grip_width_m:={target:.6f} on every trial after this one.'
        )
        return self._command_width(target)

    def lookup_tcp(self, timeout_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (position, xyzw quaternion) of the TCP in the base frame.

        Polls `can_transform` rather than leaning on `lookup_transform`'s own timeout. The two are
        not equivalent in practice: DDS discovery across the Humble NUC / Kilted host boundary can
        take many seconds, and until the TF subscription is matched the buffer holds no frames at
        all, so the lookup reports "frame does not exist" -- which reads like a crashed bringup and
        is really just an unfinished handshake. Polling lets this say which it is, and log progress
        instead of failing silently for 30 s.
        """
        base, eef = self._p['base_frame'], self._p['eef_frame']
        deadline = time.monotonic() + timeout_s
        warned = False
        while time.monotonic() < deadline:
            if self._tf_buffer.can_transform(base, eef, rclpy.time.Time()):
                tf = self._tf_buffer.lookup_transform(base, eef, rclpy.time.Time())
                t, r = tf.transform.translation, tf.transform.rotation
                return np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w])
            if not warned and time.monotonic() > deadline - timeout_s + 3.0:
                self.get_logger().info(f'waiting for TF {base} -> {eef} (DDS discovery can take a few seconds)...')
                warned = True
            time.sleep(0.1)

        known = self._tf_buffer.all_frames_as_string() or '(none)'
        raise RuntimeError(
            f'TF {base} -> {eef} did not appear within {timeout_s:.0f}s.\n'
            f'Frames the buffer did see:\n{known}\n'
            'If that list is empty, nothing is publishing /tf to this host -- check that the '
            'NUC bringup is up. If it lists fr3_* frames, the chain is incomplete rather than '
            'absent.'
        )

    def start_position(self, measured: np.ndarray) -> np.ndarray:
        """
        Pick this trial's start point: the reference position plus a fresh random offset.

        The reference is `home_xyz` when one was supplied and the measured TCP otherwise, and it
        is logged either way so the first trial of a collection can publish the number the rest
        reuse. Offsets are uniform and independent per axis, bounded by `jitter_xyz_m`.
        """
        home = [float(v) for v in self._p['home_xyz']]
        if len(home) == 3 and any(home):
            reference = np.array(home)
        else:
            if len(home) not in (0, 3):
                self.get_logger().warn(f'home_xyz needs 3 values, got {len(home)} -- ignoring it.')
            reference = np.asarray(measured, dtype=float)
        self.get_logger().info(f'HOME_XYZ={reference[0]:.6f},{reference[1]:.6f},{reference[2]:.6f}')

        bound = np.abs(np.array([float(v) for v in self._p['jitter_xyz_m']], dtype=float))
        if bound.shape != (3,):
            self.get_logger().warn(f'jitter_xyz_m needs 3 values, got {bound.size} -- not jittering.')
            return reference
        seed = int(self._p['jitter_seed'])
        rng = np.random.default_rng(None if seed < 0 else seed)
        offset = rng.uniform(-bound, bound)
        if bound.any():
            self.get_logger().info(
                f'start offset {1000 * offset[0]:+.0f},{1000 * offset[1]:+.0f},'
                f'{1000 * offset[2]:+.0f} mm from the reference '
                f'(bounds +/-{1000 * bound[0]:.0f},{1000 * bound[1]:.0f},{1000 * bound[2]:.0f} mm)'
            )
        return reference + offset

    def build_poses(self, position: np.ndarray, quat: np.ndarray) -> list[Pose]:
        """
        Build the full commanded path: move to this trial's start while levelling, settle, shake.

        The levelling phase does double duty as the approach: the tool rotates level and the TCP
        travels from where it is to the jittered start over the same `level_time_s`, so adding the
        jitter costs no extra time. Once there the position is held for the settle, and only the
        shake moves it, along `shake_axis`.
        """
        dt = self._p['waypoint_dt']
        level_quat = level_rotation(Rotation.from_quat(quat).as_matrix()[:, 2]).as_quat()
        start = self.start_position(position)

        n_level = max(2, int(round(self._p['level_time_s'] / dt)))
        travel = float(np.linalg.norm(start - np.asarray(position, dtype=float)))
        if travel > 0 and travel / max(self._p['level_time_s'], 1e-6) > MAX_SPEED_MPS:
            self.get_logger().warn(
                f'approaching the start needs {travel * 100:.1f}cm in {self._p["level_time_s"]:.1f}s '
                f'-- above the controller max of {MAX_SPEED_MPS} m/s, so it will clamp. Lengthen '
                'level_time_s or shrink jitter_xyz_m.'
            )
        ramp = np.linspace(0.0, 1.0, n_level)[:, None]
        path = np.asarray(position, dtype=float) + ramp * (start - np.asarray(position, dtype=float))
        poses = [_pose(p, q) for p, q in zip(path, slerp_quats(quat, level_quat, n_level))]
        poses += [_pose(start, level_quat)] * max(0, int(round(self._p['settle_s'] / dt)))
        position = start

        for d in shake_heights(
            self._p['amplitude_m'],
            int(self._p['n_shakes']),
            self._p['period_s'],
            dt,
            symmetric=bool(self._p['symmetric']),
        ):
            poses.append(_pose(position + self._axis * d, level_quat))
        return poses

    def report_dynamics(self) -> None:
        """
        Log what this shake asks of the arm, and warn if it is more than the controller allows.

        Worth computing rather than guessing: peak speed scales as A/T and peak acceleration as
        A/T^2, so halving the period costs four times the acceleration. The surface-tilt figure is
        the one that predicts sloshing -- a horizontal acceleration tilts effective gravity by
        atan(a/g), and it is that tilt, not the travel, that moves water.
        """
        a_m = float(self._p['amplitude_m'])
        t_s = float(self._p['period_s'])
        speed = a_m * math.pi / t_s
        accel = a_m * 2.0 * math.pi**2 / t_s**2
        force = PAYLOAD_KG * accel
        horizontal = float(np.hypot(self._axis[0], self._axis[1]))
        tilt = math.degrees(math.atan(accel * horizontal / GRAVITY))
        per_cycle = t_s / float(self._p['waypoint_dt'])

        self.get_logger().info(
            f'shake: {self._p["n_shakes"]} x {t_s:.2f}s at {a_m * 100:.1f}cm along '
            f'[{self._axis[0]:.2f} {self._axis[1]:.2f} {self._axis[2]:.2f}]'
            f'{" (symmetric)" if self._p["symmetric"] else ""} -> '
            f'peak {speed:.2f} m/s, {accel:.1f} m/s^2, {force:.1f} N, '
            f'effective-gravity tilt {tilt:.0f} deg'
        )
        if speed > MAX_SPEED_MPS:
            self.get_logger().warn(
                f'peak speed {speed:.2f} m/s exceeds the controller max_pos_speed of '
                f'{MAX_SPEED_MPS} m/s -- it will clamp, so the shake will not be what you asked '
                'for. Reduce amplitude_m or lengthen period_s.'
            )
        if force > MAX_FORCE_N:
            self.get_logger().warn(
                f'peak force {force:.1f} N exceeds the controller ceiling of {MAX_FORCE_N} N '
                f'(translational_clip x stiffness) -- the arm will lag its target rather than '
                'track it.'
            )
        if per_cycle < 10:
            self.get_logger().warn(
                f'only {per_cycle:.0f} waypoints per shake at waypoint_dt={self._p["waypoint_dt"]}s'
                ' -- too few for the controller to spline cleanly. Try waypoint_dt:=0.01.'
            )
        if horizontal < 0.3:
            self.get_logger().warn(
                'shake_axis is near-vertical, which barely sloshes: vertical acceleration only '
                'modulates effective gravity, it does not tilt it. Try shake_axis:=[1.0,0.0,0.0].'
            )

    def run_once(self) -> int:
        """Look up the TCP, build the path, publish it. Returns the waypoint count."""
        # Squeeze BEFORE the motion: the grip has to be established while the arm is still, or
        # the first upward stroke is what discovers the bottle was loose.
        if self._p['grip'] and self._p['execute']:
            self.squeeze()
            time.sleep(float(self._p['grip_time_s']))
        elif self._p['grip']:
            self.get_logger().info('dry run: not squeezing the gripper either.')

        self.report_dynamics()
        position, quat = self.lookup_tcp()
        poses = self.build_poses(position, quat)

        preview = PoseArray()
        preview.header.frame_id = self._p['base_frame']
        preview.header.stamp = self.get_clock().now().to_msg()
        preview.poses = poses
        self._preview.publish(preview)

        span = len(poses) * self._p['waypoint_dt']
        if self._pub is None:
            self.get_logger().warn(
                f'DRY RUN — {len(poses)} waypoints ({span:.1f}s) on {PREVIEW_TOPIC}. '
                'Nothing was commanded. Re-run with -p execute:=true to move the arm.'
            )
        else:
            self._pub.publish(poses, dt=self._p['waypoint_dt'])
            self.get_logger().warn(
                f'MOVING THE ARM — {len(poses)} waypoints ({span:.1f}s), '
                f'{self._p["n_shakes"]} shakes of {self._p["amplitude_m"] * 100:.0f} cm. '
                f'{CONSUMER_HINT}'
            )
        return len(poses)


def main():
    """Publish one shake sequence and exit."""
    rclpy.init()
    node = WaterShakeNode()
    # The node has to be SPINNING while lookup_transform waits, or its timeout is dead time: the
    # TF listener's subscriptions are only serviced by an executor, so a synchronous lookup on an
    # unspun node waits the full timeout and then reports the frame as nonexistent -- which reads
    # exactly like a crashed bringup and is not one. Discovery across the Humble/Kilted rmw
    # boundary takes a few seconds on its own, so this margin is the difference between working
    # and a misleading error.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.run_once()
        time.sleep(0.5)  # let the published chunk actually go out before shutdown
    except Exception as exc:  # noqa: BLE001 - a CLI tool should print, not traceback
        node.get_logger().error(str(exc))
        return 1
    finally:
        # Shut the executor down and JOIN before destroying the node: a daemon thread still inside
        # executor.spin() when the node is destroyed aborts the process with "terminate called
        # without an active exception", which buries whatever the real error was.
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
