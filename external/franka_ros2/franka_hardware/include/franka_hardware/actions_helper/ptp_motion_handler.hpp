// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <future>
#include <map>
#include <memory>

#include <franka/async_control/async_position_control_handler.hpp>
#include <franka_hardware/robot.hpp>
#include <franka_msgs/action/ptp_motion.hpp>

namespace franka_hardware {

/**
 * Handler for point-to-point motions using asynchronous position control.
 */
class PTPMotionHandler {
 public:
  // This structure holds the result after starting a new PTP motion.
  struct CommandResult {
    std::string motion_id;
    std::shared_ptr<franka_msgs::action::PTPMotion::Result> result;
  };

  /**
   * Constructs a new PTPMotionHandler with storing the franka hardware robot backend.
   *
   * @param robot the franka hardware robot backend
   */
  PTPMotionHandler(const std::shared_ptr<Robot>& robot);
  virtual ~PTPMotionHandler();

  /**
   * Starts a new point-to-point motion to the specified joint configuration.
   *
   * @param robot the (libfranka) robot backend
   * @param goal the goal containing the target joint configuration and motion parameters
   * @return CommandResult contains the result of the commanded point-to-point motion
   */
  auto startNewPTPMotion(const std::shared_ptr<franka::Robot>& robot,
                         const std::shared_ptr<const franka_msgs::action::PTPMotion::Goal>& goal)
      -> CommandResult;

  /**
   * Retrieves feedback for the specified motion ID.
   *
   * @param motion_id the ID of the motion to get feedback for
   * @return TargetFeedback containing the current feedback of the motion
   */
  auto getFeedback(const std::string& motion_id)
      -> franka::AsyncPositionControlHandler::TargetFeedback;

  /**
   * Cancels the currently running motion.
   */
  auto cancelMotion() -> void;

 private:
  std::shared_ptr<Robot> franka_hardware_robot_;

  std::shared_ptr<franka::AsyncPositionControlHandler> position_control_handler_;
  size_t motion_id_ = 0;
  std::atomic<bool> running_ = true;
  std::map<std::string, std::future<franka::AsyncPositionControlHandler::TargetFeedback>>
      feedback_futures_;

  std::mutex control_mutex_;
  franka::AsyncPositionControlHandler::TargetFeedback last_feedback_;

  /**
   * Executes the point-to-point motion to the specified joint configuration.
   *
   * @param goal_joint_configuration the target joint configuration for the motion
   * @return tuple containing the motion ID and optional TargetFeedback
   */
  auto executeMotion(const std::vector<double>& goal_joint_configuration)
      -> std::tuple<std::string,
                    std::optional<franka::AsyncPositionControlHandler::TargetFeedback>>;
};

}  // namespace franka_hardware
