# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS.JacobianTest do
  use ExUnit.Case, async: true

  alias BB.IK.DLS.Jacobian
  alias BB.IK.TestRobots.{SixDofArm, ThreeLinkArm, TwoLinkArm}

  describe "compute/4 with TwoLinkArm" do
    test "returns 3xN tensor" do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      joint_names = [:shoulder_joint, :elbow_joint]

      jacobian = Jacobian.compute(robot, positions, :tip, joint_names)

      assert Nx.shape(jacobian) == {3, 2}
      assert Nx.type(jacobian) == {:f, 64}
    end

    test "produces correct Jacobian for straight arm configuration" do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      joint_names = [:shoulder_joint, :elbow_joint]

      jacobian = Jacobian.compute(robot, positions, :tip, joint_names)

      [[j11, j12], [j21, j22], [j31, j32]] =
        jacobian |> Nx.to_list()

      assert_in_delta j11, 0.0, 0.01
      assert_in_delta j21, 0.5, 0.01

      assert_in_delta j12, 0.0, 0.01
      assert_in_delta j22, 0.2, 0.01

      assert_in_delta j31, 0.0, 0.01
      assert_in_delta j32, 0.0, 0.01
    end

    test "Jacobian changes with joint configuration" do
      robot = TwoLinkArm.robot()
      joint_names = [:shoulder_joint, :elbow_joint]

      positions1 = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      jacobian1 = Jacobian.compute(robot, positions1, :tip, joint_names)

      positions2 = %{shoulder_joint: :math.pi() / 4, elbow_joint: :math.pi() / 4}
      jacobian2 = Jacobian.compute(robot, positions2, :tip, joint_names)

      refute Nx.to_list(jacobian1) == Nx.to_list(jacobian2)
    end
  end

  describe "compute/4 with ThreeLinkArm" do
    test "returns 3x3 tensor for 3-DOF arm" do
      robot = ThreeLinkArm.robot()
      positions = %{joint1: 0.0, joint2: 0.0, joint3: 0.0}
      joint_names = [:joint1, :joint2, :joint3]

      jacobian = Jacobian.compute(robot, positions, :tip, joint_names)

      assert Nx.shape(jacobian) == {3, 3}
    end
  end

  describe "compute_with_orientation/4" do
    test "returns 6xN tensor" do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      joint_names = [:shoulder_joint, :elbow_joint]

      jacobian = Jacobian.compute_with_orientation(robot, positions, :tip, joint_names)

      assert Nx.shape(jacobian) == {6, 2}
      assert Nx.type(jacobian) == {:f, 64}
    end

    test "first 3 rows match position-only Jacobian" do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      joint_names = [:shoulder_joint, :elbow_joint]

      pos_jacobian = Jacobian.compute(robot, positions, :tip, joint_names)
      full_jacobian = Jacobian.compute_with_orientation(robot, positions, :tip, joint_names)

      pos_part = Nx.slice(full_jacobian, [0, 0], [3, 2])

      pos_list = Nx.to_flat_list(pos_jacobian)
      full_pos_list = Nx.to_flat_list(pos_part)

      Enum.zip(pos_list, full_pos_list)
      |> Enum.each(fn {a, b} ->
        assert_in_delta a, b, 1.0e-10
      end)
    end

    test "returns 6x6 tensor for 6-DOF arm" do
      robot = SixDofArm.robot()

      positions = %{
        shoulder_yaw: 0.0,
        shoulder_pitch: 0.0,
        shoulder_roll: 0.0,
        elbow_pitch: 0.0,
        wrist_pitch: 0.0,
        wrist_roll: 0.0
      }

      joint_names = [
        :shoulder_yaw,
        :shoulder_pitch,
        :shoulder_roll,
        :elbow_pitch,
        :wrist_pitch,
        :wrist_roll
      ]

      jacobian = Jacobian.compute_with_orientation(robot, positions, :tip, joint_names)

      assert Nx.shape(jacobian) == {6, 6}
    end
  end
end
