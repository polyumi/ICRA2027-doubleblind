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

#include <chrono>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <franka_msgs/action/ptp_motion.hpp>

namespace franka_example_controllers {

/**
 * The example node demonstrates point-to-point (PTP) motions using the PTP motion action server.
 *
 * The PTP motion action provides two aspects:
 * 1. Sending a goal to move the robot to a specified joint configuration with defined maximum
 *    joint velocities and goal tolerance.
 * 2. Receiving feedback during the motion and the final result upon completion.
 *
 * Hereby, the `handle_feedback` method processes feedback messages, while the `handle_result`
 * method handles the final result of the action.
 */
class PTPMotionExampleNode : public rclcpp::Node {
 public:
  explicit PTPMotionExampleNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  /**
   * @brief Sends a point-to-point motion goal to the action server
   */
  auto send_goal() -> void;

 private:
  // ROS 2 action client and name of the action
  rclcpp_action::Client<franka_msgs::action::PTPMotion>::SharedPtr client_;
  std::string action_name_;

  /**
   * @brief Callback function for handling goal responses for the point-to-point motion action
   *
   * @param goal_handle The goal handle for the point-to-point motion action
   */
  auto handle_goal_response(
      const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::PTPMotion>>&
          goal_handle) -> void;

  /**
   * @brief Callback function for handling feedback for the point-to-point motion action
   *
   * @param feedback The feedback message for the point-to-point motion action
   */
  auto handle_feedback(
      const std::shared_ptr<const franka_msgs::action::PTPMotion::Feedback>& feedback) -> void;

  /**
   * @brief Callback function for handling results for the point-to-point motion action
   *
   * @param result The wrapped result for the point-to-point motion action
   */
  auto handle_result(
      const rclcpp_action::ClientGoalHandle<franka_msgs::action::PTPMotion>::WrappedResult& result)
      -> void;
};

}  // namespace franka_example_controllers
