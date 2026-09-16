#pragma once

#include <gmock/gmock.h>

#include <franka/active_control_base.h>

class MockActiveControl : public franka::ActiveControlBase {
 public:
  MOCK_METHOD((std::pair<franka::RobotState, franka::Duration>), readOnce, (), (override));
  MOCK_METHOD(void, writeOnce, (const franka::Torques&), (override));
  MOCK_METHOD(void,
              writeOnce,
              (const franka::JointPositions&, const std::optional<const franka::Torques>&),
              (override));
  MOCK_METHOD(void,
              writeOnce,
              (const franka::JointVelocities&, const std::optional<const franka::Torques>&),
              (override));
  MOCK_METHOD(void,
              writeOnce,
              (const franka::CartesianPose&, const std::optional<const franka::Torques>&),
              (override));
  MOCK_METHOD(void,
              writeOnce,
              (const franka::CartesianVelocities&, const std::optional<const franka::Torques>&),
              (override));
  MOCK_METHOD(void, writeOnce, (const franka::JointPositions&), (override));
  MOCK_METHOD(void, writeOnce, (const franka::JointVelocities&), (override));
  MOCK_METHOD(void, writeOnce, (const franka::CartesianPose&), (override));
  MOCK_METHOD(void, writeOnce, (const franka::CartesianVelocities&), (override));
};
