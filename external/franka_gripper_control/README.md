# FAULHABER Gripper Trajectory Node

This ROS 2 node controls a FAULHABER-driven gripper over CANopen using Cyclic
Synchronous Position (CSP) mode. It receives timestamped gripper-width
trajectories, linearly interpolates between their waypoints at 200 Hz by
default, and commands the gripper in millimetres.

The user-facing width convention is:

- `0.0 mm`: fully closed calibrated endpoint
- `105.0 mm`: fully open calibrated endpoint

The node starts CSP using an existing calibration file. If calibration has not
yet been performed, the node remains alive with motor tracking disabled so the
calibration service can be called.

## ROS interfaces

### Subscribed topic

| Topic | Type | Units | Description |
|---|---|---|---|
| `/polyumi/target_gripper` | `trajectory_msgs/msg/JointTrajectory` | metres in `positions` | Timestamped trajectory chunks containing the `fr3_gripper_width` joint. |

Each trajectory point's execution time is:

```text
message.header.stamp + point.time_from_start
```

If `header.stamp` is zero, receipt time is used. The joint position supplied by
ROS is converted from metres to millimetres internally. Times in a message must
be strictly increasing.

The first received trajectory begins from the gripper's current measured
width, preventing an initial jump to the recorded starting position. New chunks
replace overlapping future waypoints while retaining older non-overlapping
points.

### Published topics

| Topic | Type | Rate | Units | Description |
|---|---|---:|---|---|
| `/faulhaber_gripper/joint_states` | `sensor_msgs/msg/JointState` | CSP rate, default 200 Hz | metres | Measured gripper width in `position[0]`, named `fr3_gripper_width`. `velocity` and `effort` are intentionally empty. |
| `/faulhaber_gripper/motor_current_ma` | `std_msgs/msg/Float64` | CSP rate, default 200 Hz | mA | Measured motor current. This is not placed in `JointState.effort`. |
| `/faulhaber_gripper/waypoint_tracking_error_mm` | `std_msgs/msg/Float64` | At source waypoints only | mm | Desired source-waypoint width minus measured width when that waypoint time is crossed. Interpolated CSP samples do not generate error messages. |

### Service

| Service | Type | Description |
|---|---|---|
| `/faulhaber_gripper/calibrate` | `std_srvs/srv/Trigger` | Stops tracking, detects both mechanical hard stops twice, saves the calibration, restarts CSP, and holds the resulting current position. |

Calibration physically moves the mechanism to both hard stops. Clear the
mechanism and supervise the gripper before calling the service.

## Launching

Build and source the workspace:

```bash
cd ~/research/ros2_ws
colcon build --packages-select franka_gripper_control --symlink-install
source install/setup.bash
```

Start the node with wall-clock time:

```bash
ros2 launch franka_gripper_control faulhaber_gripper.launch.xml
```

For rosbag playback, start the node with simulated time:

```bash
ros2 launch franka_gripper_control faulhaber_gripper.launch.xml \
  use_sim_time:=true
```

Then play the unzipped rosbag directory in another sourced terminal:

```bash
ros2 bag play /path/to/rosbag_directory --clock 200
```

For live publishers that use the normal system ROS clock, leave
`use_sim_time:=false` (the default).

## Calling the calibration service

With the mechanism clear:

```bash
ros2 service call \
  /faulhaber_gripper/calibrate \
  std_srvs/srv/Trigger \
  "{}"
```

Wait for a successful response before publishing trajectories. Calibration is
saved by default to:

```text
~/.ros/faulhaber_gripper_limits.json
```

At later startups, the node loads this file and starts at the gripper's current
physical width without recalibrating.

## Launch arguments

These are XML launch arguments passed to the node's command-line parser. Except
for `use_sim_time`, they are not ROS parameters and therefore must be placed
before or as launch `name:=value` overrides—not after `--ros-args -p`.

### ROS and CAN configuration

| Argument | Default | Meaning |
|---|---:|---|
| `use_sim_time` | `false` | Use the ROS `/clock` topic. Enable for rosbag playback with `--clock`. |
| `interface` | `can0` | SocketCAN interface connected to the motor controller. |
| `node_id` | `1` | CANopen node ID, valid range 1–127. |
| `topic` | `/polyumi/target_gripper` | Input `JointTrajectory` topic. |
| `joint_name` | `fr3_gripper_width` | Joint selected from each trajectory and published in `JointState`. |
| `calibration_file` | `~/.ros/faulhaber_gripper_limits.json` | Persistent hard-stop calibration file. |
| `calibration_service` | `/faulhaber_gripper/calibrate` | Calibration service name. |

### Calibration configuration

| Argument | Default | Units | Meaning |
|---|---:|---:|---|
| `first_speed_rpm` | `60` | motor rpm | Speed of the first approach to each hard stop. |
| `second_speed_rpm` | `10` | motor rpm | Slower verification approach after backing away. |
| `current_limit_ma` | `300` | mA | Temporary motor-current limit used during calibration. |
| `stall_current_ma` | `220` | mA | Current threshold used as one condition for hard-stop detection; it must be below `current_limit_ma`. |
| `max_open_travel` | `1000000` | increments | Maximum positive travel allowed while searching for the open stop. |
| `max_close_travel` | `2000000` | increments | Maximum negative travel allowed while searching for the closed stop. |
| `homing_timeout` | `180` | s | Maximum time allowed for each hard-stop approach. |

The travel settings are safety bounds, not commanded movements. Use the
smallest limits known to cover the real mechanism stroke.

### CSP tracking configuration

| Argument | Default | Units | Meaning |
|---|---:|---:|---|
| `csp_current_limit_ma` | `400` | mA | Motor-current limit while tracking trajectories. Contact may create persistent position error while this limit protects the mechanism. |
| `csp_rate_hz` | `200` | Hz | Interpolation, command, and feedback cycle frequency. Width and current are published at this rate. |
| `delay_compensation_ms` | `13.3` | ms | Fixed look-ahead applied when sampling the buffered trajectory to compensate for measured tracking delay. Requires future waypoints to be available. |
| `max_tracking_error_mm` | `0` | mm | Following-error shutdown threshold. `0` disables the shutdown, which permits expected error during object contact. |
| `max_command_speed_rpm` | `4000` | motor rpm | Rejects incoming trajectory segments whose estimated motor speed exceeds this validation limit. It does not command this speed. |

### Output topic configuration

| Argument | Default | Meaning |
|---|---|---|
| `joint_state_topic` | `/faulhaber_gripper/joint_states` | Measured-width topic. |
| `current_topic` | `/faulhaber_gripper/motor_current_ma` | Motor-current topic. |
| `waypoint_error_topic` | `/faulhaber_gripper/waypoint_tracking_error_mm` | Source-waypoint tracking-error topic. |

List all launch arguments and their descriptions with:

```bash
ros2 launch franka_gripper_control faulhaber_gripper.launch.xml --show-args
```

## Launch override examples

Use another CAN interface and input topic:

```bash
ros2 launch franka_gripper_control faulhaber_gripper.launch.xml \
  interface:=can1 \
  topic:=/gripper/trajectory
```

Change the calibration approach speeds:

```bash
ros2 launch franka_gripper_control faulhaber_gripper.launch.xml \
  first_speed_rpm:=40 \
  second_speed_rpm:=5
```

Change the CSP rate and current limit:

```bash
ros2 launch franka_gripper_control faulhaber_gripper.launch.xml \
  csp_rate_hz:=200 \
  csp_current_limit_ma:=400
```

Enable a 5 mm software following-error shutdown:

```bash
ros2 launch franka_gripper_control faulhaber_gripper.launch.xml \
  max_tracking_error_mm:=5
```

## Inspecting the interfaces

```bash
ros2 topic echo /faulhaber_gripper/joint_states
ros2 topic echo /faulhaber_gripper/motor_current_ma
ros2 topic echo /faulhaber_gripper/waypoint_tracking_error_mm
ros2 topic hz /faulhaber_gripper/joint_states
ros2 service list | grep faulhaber_gripper
```

## Safety and shutdown behavior

- The node holds the current calibrated width until the first trajectory is received.
- An out-of-range width is clamped into `[0, --max-width-mm]`, with a throttled warning.
- Excessive estimated waypoint-to-waypoint speed causes the complete incoming trajectory message to be rejected.
- With `max_tracking_error_mm:=0`, contact-induced following error does not stop tracking; the CSP current limit remains active.
- Communication failure, excessive scheduler lateness, or another drive error requests Quick Stop and disables the drive during shutdown.
- Pressing `Ctrl+C` stops the scheduler and disables the drive.

The CAN interface must already be configured, for example:

```bash
sudo ip link set can0 up type can bitrate 500000
```
