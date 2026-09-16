#pragma once

#include <gmock/gmock.h>

#include <franka/robot.h>
#include <optional>
#include <vector>

class MockFrankaRobot : public franka::Robot {
 public:
  MOCK_METHOD(franka::RobotState, readOnce, (), (override));
  MOCK_METHOD(std::unique_ptr<franka::ActiveControlBase>, startTorqueControl, (), (override));
  MOCK_METHOD(std::unique_ptr<franka::ActiveControlBase>,
              startJointVelocityControl,
              (const research_interface::robot::Move::ControllerMode&),
              (override));
  MOCK_METHOD(std::unique_ptr<franka::ActiveControlBase>,
              startJointPositionControl,
              (const research_interface::robot::Move::ControllerMode&),
              (override));
  MOCK_METHOD(std::unique_ptr<franka::ActiveControlBase>,
              startCartesianPoseControl,
              (const research_interface::robot::Move::ControllerMode&),
              (override));
  MOCK_METHOD(std::unique_ptr<franka::ActiveControlBase>,
              startCartesianVelocityControl,
              (const research_interface::robot::Move::ControllerMode&),
              (override));
  MOCK_METHOD(std::unique_ptr<franka::ActiveControlBase>,
              startAsyncJointPositionControl,
              (const research_interface::robot::Move::ControllerMode&,
               (const std::optional<std::vector<double>>&)),
              (override));
};
