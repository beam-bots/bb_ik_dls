# SPDX-FileCopyrightText: 2026 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS.TrackerTest do
  use ExUnit.Case, async: true

  alias BB.IK.DLS.Tracker
  alias BB.Math.Vec3
  alias BB.Robot.Kinematics
  alias BB.Robot.Runtime

  defmodule TrackerTestRobot do
    @moduledoc false
    use BB
    import BB.Unit

    settings do
      name(:dls_tracker_test_robot)
    end

    topology do
      link :base_link do
        joint :shoulder_joint do
          type(:revolute)

          axis do
          end

          limit do
            lower(~u(-180 degree))
            upper(~u(180 degree))
            effort(~u(10 newton_meter))
            velocity(~u(90 degree_per_second))
          end

          actuator(:shoulder_servo, BB.IK.TestRobots.MockActuator)

          sensor(
            :shoulder_position,
            {BB.Sensor.OpenLoopPositionEstimator, actuator: :shoulder_servo}
          )

          link :link1 do
            joint :elbow_joint do
              type(:revolute)

              origin do
                x(~u(0.3 meter))
              end

              axis do
              end

              limit do
                lower(~u(-135 degree))
                upper(~u(135 degree))
                effort(~u(5 newton_meter))
                velocity(~u(90 degree_per_second))
              end

              actuator(:elbow_servo, BB.IK.TestRobots.MockActuator)

              sensor(
                :elbow_position,
                {BB.Sensor.OpenLoopPositionEstimator, actuator: :elbow_servo}
              )

              link :link2 do
                joint :tip_joint do
                  type(:fixed)

                  origin do
                    x(~u(0.2 meter))
                  end

                  link(:tip)
                end
              end
            end
          end
        end
      end
    end
  end

  describe "start_link/1" do
    test "starts a tracker process" do
      start_supervised!(TrackerTestRobot)

      assert {:ok, pid} =
               Tracker.start_link(
                 robot: TrackerTestRobot,
                 target_link: :tip,
                 source_link: :base_link,
                 initial_target: Vec3.new(0.35, 0.2, 0.0)
               )

      assert Process.alive?(pid)
      Tracker.stop(pid)
    end
  end

  describe "stop/2" do
    test "returns the configuration the last solve arrived at, not the robot's" do
      start_supervised!(TrackerTestRobot)

      {:ok, pid} =
        Tracker.start_link(
          robot: TrackerTestRobot,
          target_link: :tip,
          source_link: :base_link,
          initial_target: Vec3.new(0.35, 0.2, 0.0)
        )

      Process.sleep(100)

      assert {:ok, solved} = Tracker.stop(pid)

      # The mock actuators discard their commands and publish no motion for the
      # estimators to interpolate, so the robot has not moved and its own
      # configuration is still the one it booted in. Reading `stop/1`'s answer
      # off the robot - as the tracker used to - would therefore hand back these
      # zeroes rather than anything the solver worked out.
      measured = Runtime.configurations(TrackerTestRobot)
      assert Enum.all?(Map.values(measured), &(&1 == 0.0))

      assert solved.shoulder_joint != 0.0
      assert solved.elbow_joint != 0.0

      # And it really is a solution, not just a non-zero pair: the tracked link
      # sits on the target when the robot is in the configuration handed back.
      {x, y, z} = Kinematics.link_position(TrackerTestRobot.robot(), solved, :tip)
      assert Vec3.distance(Vec3.new(x, y, z), Vec3.new(0.35, 0.2, 0.0)) < 0.01
    end
  end
end
