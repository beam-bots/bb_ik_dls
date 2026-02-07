# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS do
  @moduledoc """
  Damped Least Squares (Levenberg-Marquardt) inverse kinematics solver.

  DLS is a Jacobian-based iterative solver that computes joint angles to
  position an end-effector at a target pose. The damping factor prevents
  instability near kinematic singularities.

  ## Features

  - Works with `BB.Robot.State` or plain position maps
  - Position and orientation solving (quaternion or axis constraints)
  - Numerical Jacobian computation (works with any kinematic chain)
  - Adaptive damping for improved convergence
  - Respects joint limits by clamping solved values
  - Uses Nx tensors for efficient computation
  - Returns best-effort positions even on failure

  ## Usage

      robot = MyRobot.robot()
      {:ok, state} = BB.Robot.State.new(robot)

      # Solve for end-effector to reach target position
      target = Vec3.new(0.4, 0.2, 0.1)

      case BB.IK.DLS.solve(robot, state, :end_effector, target) do
        {:ok, positions, meta} ->
          BB.Robot.State.set_positions(state, positions)
          IO.puts("Solved in \#{meta.iterations} iterations")

        {:error, %BB.Error.Kinematics.NoSolution{residual: residual}} ->
          IO.puts("Failed to converge, residual: \#{residual}m")
      end

  ## Target Formats

  - `Vec3.t()` - Position-only target
  - `Transform.t()` - Position + full orientation from transform
  - `{Vec3.t(), {:quaternion, Quaternion.t()}}` - Position + explicit quaternion
  - `{Vec3.t(), {:axis, Vec3.t()}}` - Position + tool axis direction constraint

  ## Options

  - `:max_iterations` - Maximum solver iterations (default: 100)
  - `:tolerance` - Position convergence tolerance in metres (default: 1.0e-4)
  - `:orientation_tolerance` - Orientation convergence in radians (default: 0.01)
  - `:lambda` - Damping factor (default: 0.5)
  - `:adaptive_damping` - Adjust lambda based on error reduction (default: true)
  - `:step_size` - Maximum joint update per iteration in radians (default: 0.1)
  - `:respect_limits` - Whether to clamp to joint limits (default: true)
  - `:exclude_joints` - List of joint names to exclude from IK solving (default: []).
    Useful for excluding end-effectors like grippers that shouldn't affect positioning.
  - `:check_collisions` - Whether to verify the solution doesn't cause self-collision (default: false)
  - `:collision_margin` - Safety margin for collision checking in metres (default: 0.0)

  ## Algorithm

  DLS uses the damped pseudoinverse to compute joint updates:

      Δθ = J^T (J J^T + λ²I)^(-1) e

  where J is the Jacobian, e is the pose error, and λ is the damping factor.
  The damping prevents instability when J J^T is near-singular.

  ## Comparison with FABRIK

  - **DLS**: Better orientation control, handles 6-DOF arms well, slower per iteration
  - **FABRIK**: Faster for position-only, struggles with orientation, simpler algorithm
  """

  @behaviour BB.IK.Solver

  alias BB.Collision
  alias BB.Error.Kinematics.NoDofs
  alias BB.Error.Kinematics.NoSolution
  alias BB.Error.Kinematics.SelfCollision
  alias BB.Error.Kinematics.UnknownLink
  alias BB.IK.DLS.Algorithm
  alias BB.Math.Quaternion
  alias BB.Math.Transform
  alias BB.Math.Vec3
  alias BB.Robot
  alias BB.Robot.{Joint, Kinematics, State}

  @default_max_iterations 100
  @default_tolerance 1.0e-4
  @default_orientation_tolerance 0.01
  @default_lambda 0.5
  @default_step_size 0.1

  @impl true
  @spec solve(Robot.t(), State.t() | map(), atom(), BB.IK.Solver.target(), keyword()) ::
          BB.IK.Solver.solve_result()
  def solve(robot, state_or_positions, target_link, target, opts \\ [])

  def solve(%Robot{} = robot, %State{} = state, target_link, target, opts) do
    positions = State.get_all_positions(state)
    solve(robot, positions, target_link, target, opts)
  end

  def solve(%Robot{} = robot, positions, target_link, target, opts) when is_map(positions) do
    excluded_joints = Keyword.get(opts, :exclude_joints, [])

    with {:ok, joint_names} <- extract_chain_joints(robot, target_link, excluded_joints) do
      do_solve(robot, positions, target_link, target, joint_names, opts)
    end
  end

  defp do_solve(robot, positions, target_link, target, joint_names, opts) do
    config = build_config(opts)
    algorithm_config = build_algorithm_config(config)
    {target_position, orientation_target} = normalize_target(target)

    target_orientation =
      resolve_orientation_target(robot, positions, target_link, orientation_target)

    case Algorithm.iterate(
           robot,
           positions,
           target_link,
           target_position,
           target_orientation,
           joint_names,
           algorithm_config
         ) do
      {:ok, solved_positions, iteration_meta} ->
        final_positions =
          if config.respect_limits do
            clamp_to_limits(robot, solved_positions, joint_names)
          else
            solved_positions
          end

        merged_positions = Map.merge(positions, final_positions)

        with :ok <- check_collisions(robot, merged_positions, config) do
          residual = compute_residual(robot, merged_positions, target_link, target_position)

          orientation_residual =
            maybe_compute_orientation_residual(
              robot,
              merged_positions,
              target_link,
              target_orientation
            )

          meta = %{
            iterations: iteration_meta.iterations,
            residual: residual,
            orientation_residual: orientation_residual,
            reached: iteration_meta.converged
          }

          {:ok, merged_positions, meta}
        end

      {:error, :max_iterations, iteration_meta} ->
        final_positions =
          if config.respect_limits do
            clamp_to_limits(robot, iteration_meta.positions, joint_names)
          else
            iteration_meta.positions
          end

        merged_positions = Map.merge(positions, final_positions)
        residual = compute_residual(robot, merged_positions, target_link, target_position)

        {:error,
         %NoSolution{
           target_link: target_link,
           target_pose: target,
           iterations: iteration_meta.iterations,
           residual: residual,
           positions: merged_positions
         }}
    end
  end

  @doc """
  Solve IK and update the state in-place.

  Convenience function that calls `solve/5` and applies the result
  to the given `BB.Robot.State`.

  ## Returns

  Same as `solve/5`, but on success the state's ETS table is updated.
  """
  @spec solve_and_update(Robot.t(), State.t(), atom(), BB.IK.Solver.target(), keyword()) ::
          BB.IK.Solver.solve_result()
  def solve_and_update(%Robot{} = robot, %State{} = state, target_link, target, opts \\ []) do
    case solve(robot, state, target_link, target, opts) do
      {:ok, positions, meta} ->
        State.set_positions(state, positions)
        {:ok, positions, meta}

      {:error, _error} = error ->
        error
    end
  end

  defp build_config(opts) do
    %{
      max_iterations: Keyword.get(opts, :max_iterations, @default_max_iterations),
      tolerance: Keyword.get(opts, :tolerance, @default_tolerance),
      orientation_tolerance:
        Keyword.get(opts, :orientation_tolerance, @default_orientation_tolerance),
      lambda: Keyword.get(opts, :lambda, @default_lambda),
      adaptive_damping: Keyword.get(opts, :adaptive_damping, true),
      step_size: Keyword.get(opts, :step_size, @default_step_size),
      respect_limits: Keyword.get(opts, :respect_limits, true),
      check_collisions: Keyword.get(opts, :check_collisions, false),
      collision_margin: Keyword.get(opts, :collision_margin, 0.0)
    }
  end

  defp build_algorithm_config(config) do
    Map.take(config, [
      :max_iterations,
      :tolerance,
      :orientation_tolerance,
      :lambda,
      :adaptive_damping,
      :step_size,
      :respect_limits
    ])
  end

  defp extract_chain_joints(robot, target_link, excluded_joints) do
    case Robot.path_to(robot, target_link) do
      nil -> {:error, %UnknownLink{target_link: target_link}}
      path -> filter_movable_joints(robot, path, target_link, excluded_joints)
    end
  end

  defp filter_movable_joints(robot, path, target_link, excluded_joints) do
    joint_names = Enum.filter(path, &movable_joint?(robot, &1, excluded_joints))

    if Enum.empty?(joint_names) do
      {:error, %NoDofs{target_link: target_link}}
    else
      {:ok, joint_names}
    end
  end

  defp movable_joint?(robot, name, excluded_joints) do
    case Map.get(robot.joints, name) do
      nil -> false
      joint -> Joint.movable?(joint) and name not in excluded_joints
    end
  end

  defp normalize_target(%Vec3{} = vec) do
    {vec, :none}
  end

  defp normalize_target({%Vec3{} = vec, orientation}) do
    {vec, normalize_orientation(orientation)}
  end

  defp normalize_target(%Transform{} = transform) do
    pos_vec = Transform.get_translation(transform)
    orientation = {:quaternion, Transform.get_quaternion(transform)}
    {pos_vec, orientation}
  end

  defp normalize_orientation(:none), do: :none
  defp normalize_orientation({:axis, %Vec3{} = vec}), do: {:axis, vec}
  defp normalize_orientation({:quaternion, %Quaternion{} = q}), do: {:quaternion, q}

  defp resolve_orientation_target(_robot, _positions, _target_link, :none), do: nil

  defp resolve_orientation_target(_robot, _positions, _target_link, {:quaternion, q}), do: q

  defp resolve_orientation_target(robot, positions, target_link, {:axis, axis_vec}) do
    current_transform = Kinematics.forward_kinematics(robot, positions, target_link)
    current_quat = Transform.get_quaternion(current_transform)

    current_z = Quaternion.rotate_vector(current_quat, Vec3.unit_z())
    target_axis = Vec3.normalise(axis_vec)

    rotation = Quaternion.from_two_vectors(current_z, target_axis)
    Quaternion.multiply(rotation, current_quat)
  end

  defp clamp_to_limits(robot, positions, joint_names) do
    Enum.reduce(joint_names, positions, fn joint_name, acc ->
      joint = Map.get(robot.joints, joint_name)
      position = Map.get(acc, joint_name, 0.0)

      clamped =
        case joint.limits do
          nil ->
            position

          %{lower: nil, upper: nil} ->
            position

          %{lower: lower, upper: nil} ->
            max(position, lower)

          %{lower: nil, upper: upper} ->
            min(position, upper)

          %{lower: lower, upper: upper} ->
            position |> max(lower) |> min(upper)
        end

      Map.put(acc, joint_name, clamped)
    end)
  end

  defp compute_residual(robot, positions, target_link, target_position) do
    {x, y, z} = Kinematics.link_position(robot, positions, target_link)
    actual = Vec3.new(x, y, z)

    Vec3.subtract(actual, target_position)
    |> Vec3.magnitude()
  end

  defp compute_orientation_residual(robot, positions, target_link, target_quaternion) do
    current_transform = Kinematics.forward_kinematics(robot, positions, target_link)
    current_quat = Transform.get_quaternion(current_transform)

    Quaternion.angular_distance(current_quat, target_quaternion)
  end

  defp maybe_compute_orientation_residual(_robot, _positions, _target_link, nil), do: nil

  defp maybe_compute_orientation_residual(robot, positions, target_link, target_orientation) do
    compute_orientation_residual(robot, positions, target_link, target_orientation)
  end

  defp check_collisions(_robot, _positions, %{check_collisions: false}), do: :ok

  defp check_collisions(robot, positions, %{check_collisions: true, collision_margin: margin}) do
    case Collision.detect_self_collisions(robot, positions, margin: margin) do
      [] ->
        :ok

      [collision | _] ->
        {:error,
         %SelfCollision{
           link_a: collision.link_a,
           link_b: collision.link_b,
           joint_positions: positions
         }}
    end
  end
end
