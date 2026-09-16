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

#include "franka_example_controllers/async_motions/ptp_motion_example_node.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <franka_msgs/action/ptp_motion.hpp>

using namespace std::chrono_literals;

const std::vector<double> kInitialGoalJointConfiguration_ = {-0.5, -M_PI_4, 0,     -3 * M_PI_4,
                                                             0,    M_PI_2,  M_PI_4};
const std::vector<double> kMaximumJointVelocities_ = std::vector<double>(7, 1.0);
constexpr double kGoalTolerance_{0.01};

namespace franka_example_controllers {

PTPMotionExampleNode::PTPMotionExampleNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("ptp_motion_example", options) {
  action_name_ = "action_server/ptp_motion";

  client_ = rclcpp_action::create_client<franka_msgs::action::PTPMotion>(this, action_name_);

  if (!client_->wait_for_action_server(5s)) {
    RCLCPP_ERROR(get_logger(), "Action server '%s' not available.", action_name_.c_str());
    return;
  }
}

auto PTPMotionExampleNode::handle_goal_response(
    const std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::PTPMotion>>&
        goal_handle) -> void {
  if (!goal_handle) {
    RCLCPP_ERROR(get_logger(), "PTP Motion goal was rejected.");
  } else {
    RCLCPP_INFO(get_logger(), "PTP Motion goal accepted.");
  }
}

auto PTPMotionExampleNode::handle_feedback(
    const std::shared_ptr<const franka_msgs::action::PTPMotion::Feedback>& feedback) -> void {
  static int count = 0;
  if (count++ % 100 == 0) {
    RCLCPP_INFO(get_logger(), "PTP Motion feedback status: %u", feedback->target_status.status);
  }
}

auto PTPMotionExampleNode::handle_result(
    const rclcpp_action::ClientGoalHandle<franka_msgs::action::PTPMotion>::WrappedResult& result)
    -> void {
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      RCLCPP_INFO(get_logger(), "PTP Motion succeeded.");
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(get_logger(), "PTP Motion aborted: %s", result.result->error_message.c_str());
      break;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_WARN(get_logger(), "PTP Motion canceled.");
      break;
    default:
      RCLCPP_ERROR(get_logger(), "PTP Motion unknown result code.");
      break;
  }
}

void PTPMotionExampleNode::send_goal() {
  // Define the goal
  franka_msgs::action::PTPMotion::Goal goal;
  goal.goal_joint_configuration = kInitialGoalJointConfiguration_;
  goal.maximum_joint_velocities = kMaximumJointVelocities_;
  goal.goal_tolerance = kGoalTolerance_;

  // Set up all callback functions for the action client
  rclcpp_action::Client<franka_msgs::action::PTPMotion>::SendGoalOptions options;
  options.goal_response_callback =
      [this](
          std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::PTPMotion>> handle) {
        handle_goal_response(handle);
      };

  options.feedback_callback =
      [this](std::shared_ptr<rclcpp_action::ClientGoalHandle<franka_msgs::action::PTPMotion>>,
             const std::shared_ptr<const franka_msgs::action::PTPMotion::Feedback>& feedback) {
        handle_feedback(feedback);
      };

  std::atomic<bool> motion_finished{false};
  options.result_callback =
      [&](const rclcpp_action::ClientGoalHandle<franka_msgs::action::PTPMotion>::WrappedResult&
              result) {
        handle_result(result);

        // Motion is finished - with or without error
        motion_finished.store(true);
      };

  // Send the first goal to the action server. The final result will be handled in the result
  // callback and set the 'motion_finished' flag if the motion reached its destination or was
  // aborted.
  RCLCPP_INFO(get_logger(), "Sending initial PTP Motion goal.");
  auto future_handle = client_->async_send_goal(goal, options);
  if (!future_handle.valid()) {
    RCLCPP_ERROR(get_logger(), "Failed to send PTP Motion goal.");
  } else {
    RCLCPP_INFO(get_logger(), "PTP Motion goal sents.");
  }

  auto goal_handle = future_handle.get();
  if (!goal_handle) {
    RCLCPP_ERROR(get_logger(), "PTP Motion goal was rejected by server.");
    return;
  }

  while (!motion_finished.load()) {
    rclcpp::sleep_for(100ms);
  }
  motion_finished.store(false);

  // Execute multiple motions in alternating directions. Note, we always reset 'motion_finished'
  // back to 'false' before sending a new goal. Reaching the goal or an abort will set it back to
  // 'true' in the result callback.
  RCLCPP_INFO(get_logger(), "Sending new PTP Motion goals in alternating directions.");

  auto direction = 1;
  auto max_repetitions = 20U;
  for (size_t counter = 0; counter < max_repetitions; ++counter) {
    if (counter == max_repetitions - 1) {
      goal.goal_joint_configuration[0] = 0;
      goal.goal_joint_configuration[2] = 0;
    } else {
      goal.goal_joint_configuration[0] = direction * 1.0 * (counter + 1) / 10.0;
      goal.goal_joint_configuration[2] = direction * 0.75 * (counter + 1) / 10.0;
    }
    direction *= -1;

    RCLCPP_INFO(get_logger(), "Sending repeat PTP Motion goal %ld.", counter + 1);
    future_handle = client_->async_send_goal(goal, options);

    while (!motion_finished.load()) {
      rclcpp::sleep_for(100ms);
    }
    motion_finished.store(false);
  }

  RCLCPP_INFO(get_logger(), "PTP Motion example completed.");
}

}  // namespace franka_example_controllers

int main(int argc, char** argv) {
  // Initialize the ROS 2 node and prepare the action client
  rclcpp::init(argc, argv);
  auto node = std::make_shared<franka_example_controllers::PTPMotionExampleNode>();
  auto send_goal_future = std::async(std::launch::async, [&node]() { node->send_goal(); });

  while (send_goal_future.wait_for(1ms) != std::future_status::ready) {
    // While waiting for the termination of the motions, we need to spin the node to process
    // incoming messages (e.g., feedback and result from the action server)
    rclcpp::spin(node);
  }

  rclcpp::shutdown();
  return 0;
}
