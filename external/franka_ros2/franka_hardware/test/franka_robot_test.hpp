#include <gmock/gmock.h>
#include <algorithm>
#include <functional>

#include <franka/active_control_base.h>
#include <franka/robot.h>
#include "franka_hardware/robot.hpp"
#include "franka_hardware_mocks/franka_active_control_mock.hpp"
#include "franka_hardware_mocks/franka_model_mock.hpp"
#include "franka_hardware_mocks/franka_robot_mock.hpp"
#include "test_utils.hpp"

#pragma once

namespace franka {

template <typename T>
bool compareWithTolerance(const T& lhs, const T& rhs) {
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                    [](double lhs_element, double rhs_element) {
                      return std::abs(lhs_element - rhs_element) < k_EPS;
                    });
}

bool operator==(const CartesianVelocities& lhs, const CartesianVelocities& rhs) {
  return compareWithTolerance(lhs.O_dP_EE, rhs.O_dP_EE);
}

bool operator==(const CartesianPose& lhs, const CartesianPose& rhs) {
  return compareWithTolerance(lhs.O_T_EE, rhs.O_T_EE);
}

bool operator==(const JointPositions& lhs, const JointPositions& rhs) {
  return compareWithTolerance(lhs.q, rhs.q);
}

bool operator==(const JointVelocities& lhs, const JointVelocities& rhs) {
  return compareWithTolerance(lhs.dq, rhs.dq);
}

bool operator==(const Torques& lhs, const Torques& rhs) {
  return compareWithTolerance(lhs.tau_J, rhs.tau_J);
}
}  // namespace franka

class FrankaRobotTests : public ::testing::Test {
 protected:
  std::unique_ptr<MockFrankaRobot> mock_libfranka_robot;
  std::unique_ptr<MockModel> mock_model;
  std::unique_ptr<MockActiveControl> mock_active_control;

  template <typename RobotInitFunction, typename ControlType, typename RawControlInputType>
  void testReadWriteOnce(RobotInitFunction initFunction,
                         std::function<void()> expectCallFunction,
                         const RawControlInputType& control_input,
                         const ControlType& expected_active_control_input) {
    EXPECT_CALL(*mock_active_control, writeOnce(expected_active_control_input));
    EXPECT_CALL(*mock_active_control, readOnce());
    expectCallFunction();
    franka_hardware::Robot robot(std::move(mock_libfranka_robot), std::move(mock_model));
    (robot.*initFunction)();
    robot.readOnce();
    robot.writeOnce(control_input);
  }

  void SetUp() override {
    mock_libfranka_robot = std::make_unique<MockFrankaRobot>();
    mock_model = std::make_unique<MockModel>();
    mock_active_control = std::make_unique<MockActiveControl>();
  }
  void TearDown() override {
    mock_libfranka_robot.reset();
    mock_model.reset();
    mock_active_control.reset();
  }
};
