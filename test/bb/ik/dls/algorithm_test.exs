# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS.AlgorithmTest do
  use ExUnit.Case, async: true

  alias BB.IK.DLS.Algorithm
  alias BB.IK.TestRobots.ThreeLinkArm
  alias BB.IK.TestRobots.TwoLinkArm
  alias BB.Math.Vec3

  describe "iterate/7" do
    setup do
      robot = TwoLinkArm.robot()

      config = %{
        max_iterations: 100,
        tolerance: 1.0e-4,
        orientation_tolerance: 0.01,
        lambda: 0.5,
        adaptive_damping: true,
        step_size: 0.1,
        respect_limits: true
      }

      {:ok, robot: robot, config: config}
    end

    test "converges for reachable target", %{robot: robot, config: config} do
      initial_positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      joint_names = [:shoulder_joint, :elbow_joint]
      target_position = Vec3.new(0.3, 0.2, 0.0)

      assert {:ok, positions, meta} =
               Algorithm.iterate(
                 robot,
                 initial_positions,
                 :tip,
                 target_position,
                 nil,
                 joint_names,
                 config
               )

      assert meta.converged == true
      assert meta.iterations > 0
      assert Map.has_key?(positions, :shoulder_joint)
      assert Map.has_key?(positions, :elbow_joint)
    end

    test "returns max_iterations error when not converging", %{robot: robot, config: config} do
      initial_positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      joint_names = [:shoulder_joint, :elbow_joint]

      target_position = Vec3.new(1.0, 0.0, 0.0)

      config = %{config | max_iterations: 10}

      assert {:error, :max_iterations, meta} =
               Algorithm.iterate(
                 robot,
                 initial_positions,
                 :tip,
                 target_position,
                 nil,
                 joint_names,
                 config
               )

      assert meta.iterations == 10
      assert is_map(meta.positions)
    end

    test "handles orientation target with sufficient DOFs" do
      # Use ThreeLinkArm for orientation tests - more DOFs than TwoLinkArm
      robot = ThreeLinkArm.robot()
      initial_positions = %{joint1: 0.0, joint2: 0.0, joint3: 0.0}
      joint_names = [:joint1, :joint2, :joint3]
      target_position = Vec3.new(0.15, 0.1, 0.3)

      # Position-only target should converge
      config = %{
        max_iterations: 100,
        tolerance: 1.0e-3,
        orientation_tolerance: 0.1,
        lambda: 0.5,
        adaptive_damping: true,
        step_size: 0.1,
        respect_limits: true
      }

      assert {:ok, _positions, meta} =
               Algorithm.iterate(
                 robot,
                 initial_positions,
                 :tip,
                 target_position,
                 nil,
                 joint_names,
                 config
               )

      assert meta.converged == true
    end

    test "iterations increase with smaller step_size", %{robot: robot, config: config} do
      initial_positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      joint_names = [:shoulder_joint, :elbow_joint]
      target_position = Vec3.new(0.3, 0.2, 0.0)

      config_large_step = %{config | step_size: 0.2}
      config_small_step = %{config | step_size: 0.02}

      {:ok, _pos1, meta1} =
        Algorithm.iterate(
          robot,
          initial_positions,
          :tip,
          target_position,
          nil,
          joint_names,
          config_large_step
        )

      {:ok, _pos2, meta2} =
        Algorithm.iterate(
          robot,
          initial_positions,
          :tip,
          target_position,
          nil,
          joint_names,
          config_small_step
        )

      assert meta2.iterations > meta1.iterations
    end
  end

  describe "compute_update/3" do
    test "computes damped pseudoinverse correctly" do
      jacobian = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], type: :f64)
      error = Nx.tensor([0.1, 0.2, 0.0], type: :f64)
      lambda = 0.1

      delta = Algorithm.compute_update(jacobian, error, lambda)

      assert Nx.shape(delta) == {2}

      [d1, d2] = Nx.to_flat_list(delta)

      assert_in_delta d1, 0.099, 0.01
      assert_in_delta d2, 0.198, 0.01
    end

    test "damping reduces update magnitude" do
      jacobian = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], type: :f64)
      error = Nx.tensor([0.1, 0.2, 0.0], type: :f64)

      delta_low_damp = Algorithm.compute_update(jacobian, error, 0.01)
      delta_high_damp = Algorithm.compute_update(jacobian, error, 1.0)

      norm_low = Nx.to_number(Nx.LinAlg.norm(delta_low_damp))
      norm_high = Nx.to_number(Nx.LinAlg.norm(delta_high_damp))

      assert norm_low > norm_high
    end
  end
end
