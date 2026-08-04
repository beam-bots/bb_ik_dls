# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLSTest do
  use ExUnit.Case, async: true

  alias BB.Collision
  alias BB.Error.Kinematics.NoDofs
  alias BB.Error.Kinematics.NoSolution
  alias BB.Error.Kinematics.NotAnAncestor
  alias BB.Error.Kinematics.SelfCollision
  alias BB.Error.Kinematics.UnknownLink
  alias BB.IK.DLS

  alias BB.IK.TestRobots.{
    CollisionArm,
    FixedOnlyChain,
    FloatingBaseArm,
    PlanarBaseArm,
    PrismaticArm,
    SixDofArm,
    ThreeLinkArm,
    TwoLinkArm
  }

  alias BB.Math.Quaternion
  alias BB.Math.Transform
  alias BB.Math.Transform2D
  alias BB.Math.Vec3
  alias BB.Robot.Kinematics

  describe "solve/5 with TwoLinkArm" do
    setup do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      {:ok, robot: robot, positions: positions}
    end

    test "solves for reachable target", %{robot: robot, positions: positions} do
      target = Vec3.new(0.3, 0.2, 0.0)

      assert {:ok, solved_positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)

      assert is_map(solved_positions)
      assert Map.has_key?(solved_positions, :shoulder_joint)
      assert Map.has_key?(solved_positions, :elbow_joint)
      assert meta.iterations > 0
      assert meta.residual < 0.01
      assert meta.reached == true

      {x, y, z} = Kinematics.link_position(robot, solved_positions, :tip)
      assert_in_delta x, 0.3, 0.01
      assert_in_delta y, 0.2, 0.01
      assert_in_delta z, 0.0, 0.01
    end

    test "solves for target at extended reach", %{robot: robot, positions: positions} do
      target = Vec3.new(0.45, 0.1, 0.0)

      assert {:ok, solved_positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)

      assert meta.residual < 0.01
      assert meta.reached == true

      {x, y, _z} = Kinematics.link_position(robot, solved_positions, :tip)
      assert_in_delta x, 0.45, 0.01
      assert_in_delta y, 0.1, 0.01
    end

    test "returns error for unreachable target", %{robot: robot, positions: positions} do
      target = Vec3.new(1.0, 0.0, 0.0)

      assert {:error, %NoSolution{} = error} =
               DLS.solve(robot, positions, :base_link, :tip, target)

      assert error.target_link == :tip
      assert is_map(error.positions)
      assert error.residual > 0.4
    end

    test "returns error for unknown target link", %{robot: robot, positions: positions} do
      target = Vec3.new(0.3, 0.2, 0.0)

      assert {:error, %UnknownLink{link: :nonexistent, role: :target}} =
               DLS.solve(robot, positions, :base_link, :nonexistent, target)
    end

    test "returns error for unknown source link", %{robot: robot, positions: positions} do
      target = Vec3.new(0.3, 0.2, 0.0)

      assert {:error, %UnknownLink{link: :nonexistent, role: :source}} =
               DLS.solve(robot, positions, :nonexistent, :tip, target)
    end

    # The chain is scoped by both ends now, so asking for them the wrong way round
    # is a distinct failure that names the link the caller should have passed.
    test "returns error when the source is below the target", %{
      robot: robot,
      positions: positions
    } do
      target = Vec3.new(0.3, 0.2, 0.0)

      assert {:error, %NotAnAncestor{common_ancestor: :base_link}} =
               DLS.solve(robot, positions, :tip, :base_link, target)
    end
  end

  describe "solve/5 with FixedOnlyChain" do
    test "returns error for chain with no DOFs" do
      robot = FixedOnlyChain.robot()
      positions = %{}
      target = Vec3.new(0.0, 0.0, 0.1)

      assert {:error, %NoDofs{target_link: :end_link}} =
               DLS.solve(robot, positions, :base_link, :end_link, target)
    end
  end

  describe "solve/5 with ThreeLinkArm (3D)" do
    setup do
      robot = ThreeLinkArm.robot()
      positions = %{joint1: 0.0, joint2: 0.0, joint3: 0.0}
      {:ok, robot: robot, positions: positions}
    end

    test "solves for 3D target", %{robot: robot, positions: positions} do
      target = Vec3.new(0.15, 0.15, 0.3)

      assert {:ok, solved_positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)

      assert meta.residual < 0.02
      assert meta.reached == true

      {x, y, z} = Kinematics.link_position(robot, solved_positions, :tip)
      assert_in_delta x, 0.15, 0.02
      assert_in_delta y, 0.15, 0.02
      assert_in_delta z, 0.3, 0.02
    end
  end

  describe "solve/5 with PrismaticArm" do
    test "solves with prismatic joint" do
      robot = PrismaticArm.robot()
      positions = %{rotate_joint: 0.0, slide_joint: 0.0}

      target = Vec3.new(0.4, 0.1, 0.0)

      assert {:ok, solved_positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)

      assert meta.residual < 0.02
      assert Map.has_key?(solved_positions, :slide_joint)
    end
  end

  describe "target formats" do
    test "accepts Vec3 as target" do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      target = Vec3.new(0.3, 0.2, 0.0)

      assert {:ok, _positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)
      assert meta.orientation_residual == nil
    end

    test "accepts Transform as target" do
      # Use 6DOF arm for orientation targets
      robot = SixDofArm.robot()

      positions = %{
        shoulder_yaw: 0.0,
        shoulder_pitch: 0.0,
        shoulder_roll: 0.0,
        elbow_pitch: 0.0,
        wrist_pitch: 0.0,
        wrist_roll: 0.0
      }

      target =
        Transform.from_position_quaternion(
          Vec3.new(0.2, 0.1, 0.4),
          Quaternion.identity()
        )

      assert {:ok, _positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)
      assert is_float(meta.orientation_residual)
    end

    test "accepts {Vec3, {:quaternion, Quaternion}} tuple" do
      # Use 6DOF arm for orientation targets
      robot = SixDofArm.robot()

      positions = %{
        shoulder_yaw: 0.0,
        shoulder_pitch: 0.0,
        shoulder_roll: 0.0,
        elbow_pitch: 0.0,
        wrist_pitch: 0.0,
        wrist_roll: 0.0
      }

      target = {Vec3.new(0.2, 0.1, 0.4), {:quaternion, Quaternion.identity()}}

      assert {:ok, _positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)
      assert is_float(meta.orientation_residual)
    end

    test "accepts {Vec3, {:axis, Vec3}} tuple" do
      # Use 6DOF arm for orientation targets
      robot = SixDofArm.robot()

      positions = %{
        shoulder_yaw: 0.0,
        shoulder_pitch: 0.0,
        shoulder_roll: 0.0,
        elbow_pitch: 0.0,
        wrist_pitch: 0.0,
        wrist_roll: 0.0
      }

      target = {Vec3.new(0.2, 0.1, 0.4), {:axis, Vec3.unit_z()}}

      assert {:ok, _positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)
      assert is_float(meta.orientation_residual)
    end
  end

  describe "options" do
    setup do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}
      target = Vec3.new(0.3, 0.2, 0.0)
      {:ok, robot: robot, positions: positions, target: target}
    end

    test "respects max_iterations", %{robot: robot, positions: positions, target: target} do
      assert {:error, %NoSolution{iterations: 5}} =
               DLS.solve(robot, positions, :base_link, :tip, target, max_iterations: 5)
    end

    test "respects tolerance", %{robot: robot, positions: positions, target: target} do
      assert {:ok, _positions, meta} =
               DLS.solve(robot, positions, :base_link, :tip, target, tolerance: 0.1)

      assert meta.residual < 0.1
    end

    test "respects lambda option", %{robot: robot, positions: positions, target: target} do
      assert {:ok, _positions, _meta} =
               DLS.solve(robot, positions, :base_link, :tip, target, lambda: 1.0)
    end

    test "respects adaptive_damping option", %{robot: robot, positions: positions, target: target} do
      # Without adaptive damping, may need more iterations or higher tolerance
      assert {:ok, _positions, _meta} =
               DLS.solve(robot, positions, :base_link, :tip, target,
                 adaptive_damping: false,
                 max_iterations: 200,
                 tolerance: 0.001
               )
    end

    test "respects step_size option", %{robot: robot, positions: positions, target: target} do
      assert {:ok, _positions, _meta} =
               DLS.solve(robot, positions, :base_link, :tip, target, step_size: 0.05)
    end
  end

  describe "joint limits" do
    test "clamps joint values to limits when respect_limits: true" do
      robot = TwoLinkArm.robot()
      positions = %{shoulder_joint: 0.0, elbow_joint: 0.0}

      target = Vec3.new(-0.3, 0.2, 0.0)

      assert {:ok, solved_positions, _meta} =
               DLS.solve(robot, positions, :base_link, :tip, target, respect_limits: true)

      shoulder = Map.get(solved_positions, :shoulder_joint)
      elbow = Map.get(solved_positions, :elbow_joint)

      assert shoulder >= -:math.pi()
      assert shoulder <= :math.pi()
      assert elbow >= -135 * :math.pi() / 180
      assert elbow <= 135 * :math.pi() / 180
    end
  end

  describe "6-DOF orientation solving" do
    setup do
      robot = SixDofArm.robot()

      positions = %{
        shoulder_yaw: 0.0,
        shoulder_pitch: 0.0,
        shoulder_roll: 0.0,
        elbow_pitch: 0.0,
        wrist_pitch: 0.0,
        wrist_roll: 0.0
      }

      {:ok, robot: robot, positions: positions}
    end

    test "solves position with 6-DOF arm", %{robot: robot, positions: positions} do
      target = Vec3.new(0.2, 0.2, 0.3)

      assert {:ok, solved_positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)

      assert meta.residual < 0.02
      assert meta.orientation_residual == nil

      {x, y, z} = Kinematics.link_position(robot, solved_positions, :tip)
      assert_in_delta x, 0.2, 0.02
      assert_in_delta y, 0.2, 0.02
      assert_in_delta z, 0.3, 0.02
    end

    test "solves with quaternion orientation target", %{robot: robot, positions: positions} do
      target_pos = Vec3.new(0.2, 0.1, 0.4)
      target_quat = Quaternion.from_axis_angle(Vec3.unit_z(), :math.pi() / 4)
      target = {target_pos, {:quaternion, target_quat}}

      assert {:ok, _solved_positions, meta} =
               DLS.solve(robot, positions, :base_link, :tip, target)

      assert meta.residual < 0.05
      assert is_float(meta.orientation_residual)
    end

    test "returns orientation_residual in metadata", %{robot: robot, positions: positions} do
      target =
        Transform.from_position_quaternion(
          Vec3.new(0.2, 0.1, 0.4),
          Quaternion.identity()
        )

      assert {:ok, _positions, meta} = DLS.solve(robot, positions, :base_link, :tip, target)

      assert is_float(meta.orientation_residual)
      assert meta.orientation_residual >= 0.0
    end
  end

  describe "collision checking" do
    setup do
      robot = CollisionArm.robot()
      positions = %{shoulder: 0.0, elbow: 0.0}
      {:ok, robot: robot, positions: positions}
    end

    test "does not check collisions by default", %{robot: robot, positions: positions} do
      target = Vec3.new(0.4, 0.0, 0.2)

      assert {:ok, _solved_positions, _meta} =
               DLS.solve(robot, positions, :base_link, :tip, target)
    end

    test "succeeds when solution has no self-collision", %{robot: robot, positions: positions} do
      target = Vec3.new(0.4, 0.0, 0.2)

      assert {:ok, solved_positions, _meta} =
               DLS.solve(robot, positions, :base_link, :tip, target, check_collisions: true)

      refute Collision.self_collision?(robot, solved_positions)
    end

    test "returns SelfCollision error when collision margin triggers detection", %{
      robot: robot,
      positions: positions
    } do
      target = Vec3.new(0.4, 0.0, 0.2)

      assert {:error, %SelfCollision{} = error} =
               DLS.solve(robot, positions, :base_link, :tip, target,
                 check_collisions: true,
                 collision_margin: 1.0
               )

      assert is_atom(error.link_a)
      assert is_atom(error.link_b)
      assert is_map(error.joint_positions)
    end

    test "respects collision_margin option", %{robot: robot, positions: positions} do
      target = Vec3.new(0.35, 0.0, 0.15)

      assert {:ok, solved_positions, _meta} =
               DLS.solve(robot, positions, :base_link, :tip, target, check_collisions: true)

      refute Collision.self_collision?(robot, solved_positions)

      assert {:error, %SelfCollision{}} =
               DLS.solve(robot, positions, :base_link, :tip, target,
                 check_collisions: true,
                 collision_margin: 0.5
               )
    end
  end

  # Damped least squares is a pseudo-inverse over whatever Jacobian it is handed,
  # so a wider Jacobian should need no special case. These tests exist to prove
  # that claim rather than assume it.
  describe "chains containing a multi-DoF joint" do
    setup do
      %{
        planar: PlanarBaseArm.robot(),
        floating: FloatingBaseArm.robot()
      }
    end

    test "reports the extra degrees of freedom in its column layout", %{planar: robot} do
      assert Kinematics.jacobian_columns(robot, [:base, :shoulder_joint, :elbow_joint]) == [
               {:base, 0},
               {:base, 1},
               {:base, 2},
               {:shoulder_joint, 0},
               {:elbow_joint, 0}
             ]
    end

    test "solves a target the arm alone cannot reach, by driving the base", %{planar: robot} do
      configurations = %{base: Transform2D.identity(), shoulder_joint: 0.0}

      # The arm reaches 0.5m; this is 2m out, so the base must move.
      target = Vec3.new(2.0, 0.0, 0.0)

      assert {:ok, solved, meta} =
               DLS.solve(robot, configurations, :odom, :tip, target, max_iterations: 400)

      assert meta.reached, "expected convergence, residual was #{meta.residual}"
      assert %Transform2D{} = solved.base

      {x, y, z} = Kinematics.link_position(robot, solved, :tip)
      assert_in_delta x, 2.0, 1.0e-3
      assert_in_delta y, 0.0, 1.0e-3
      assert_in_delta z, 0.0, 1.0e-3
    end

    test "leaves the planar configuration a Transform2D, not a float", %{planar: robot} do
      configurations = %{base: Transform2D.identity(), shoulder_joint: 0.0}

      assert {:ok, solved, _meta} =
               DLS.solve(robot, configurations, :odom, :tip, Vec3.new(1.0, 0.5, 0.0),
                 max_iterations: 400
               )

      assert %Transform2D{} = solved.base
      assert is_float(solved.shoulder_joint)
    end

    test "solves through a floating base", %{floating: robot} do
      configurations = %{base: Transform.identity(), gimbal: 0.0}
      target = Vec3.new(1.0, 2.0, 3.0)

      assert {:ok, solved, meta} =
               DLS.solve(robot, configurations, :world, :lens, target, max_iterations: 400)

      assert meta.reached, "expected convergence, residual was #{meta.residual}"
      assert %Transform{} = solved.base

      {x, y, z} = Kinematics.link_position(robot, solved, :lens)
      assert_in_delta x, 1.0, 1.0e-3
      assert_in_delta y, 2.0, 1.0e-3
      assert_in_delta z, 3.0, 1.0e-3
    end

    # Scoping the chain past the multi-DoF joint is what lets a solver that cannot
    # handle one still work on such a robot.
    test "a chain scoped below the base contains no multi-DoF joint", %{planar: robot} do
      assert Kinematics.jacobian_columns(robot, [:shoulder_joint, :elbow_joint]) == [
               {:shoulder_joint, 0},
               {:elbow_joint, 0}
             ]

      configurations = %{base: Transform2D.new(1.0, 0.0, 0.0), shoulder_joint: 0.0}

      assert {:ok, solved, _meta} =
               DLS.solve(robot, configurations, :base_link, :tip, Vec3.new(1.0, 0.5, 0.0),
                 max_iterations: 400
               )

      # The base was not part of the problem, so it must come back untouched.
      assert solved.base == Transform2D.new(1.0, 0.0, 0.0)
    end
  end
end
