# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS.Motion do
  @moduledoc """
  Convenience functions for DLS-based motion.

  This module wraps `BB.Motion` with the DLS solver pre-configured,
  providing a simpler API for common inverse kinematics motion tasks.

  ## Single Target

      # Move end-effector to target position
      case BB.IK.DLS.Motion.move_to(MyRobot, :gripper, {0.3, 0.2, 0.1}) do
        {:ok, meta} -> IO.puts("Reached in \#{meta.iterations} iterations")
        {:error, reason, _meta} -> IO.puts("Failed: \#{reason}")
      end

      # Just solve without moving (for validation)
      case BB.IK.DLS.Motion.solve(MyRobot, :gripper, {0.3, 0.2, 0.1}) do
        {:ok, positions, meta} -> IO.inspect(positions)
        {:error, reason, _meta} -> IO.puts("Unreachable: \#{reason}")
      end

  ## Multiple Targets (for gait, coordinated motion)

      targets = %{left_foot: {0.1, 0.0, 0.0}, right_foot: {-0.1, 0.0, 0.0}}

      case BB.IK.DLS.Motion.move_to_multi(MyRobot, targets) do
        {:ok, results} -> IO.puts("All targets reached")
        {:error, failed, reason, _} -> IO.puts("Failed: \#{failed}: \#{reason}")
      end

  ## In Custom Commands

      def handle_command(%{target: target}, context) do
        case BB.IK.DLS.Motion.move_to(context, :gripper, target) do
          {:ok, meta} -> {:ok, %{residual: meta.residual}}
          {:error, reason, _meta} -> {:error, reason}
        end
      end
  """

  alias BB.Command.Context
  alias BB.IK.DLS
  alias BB.Motion

  @type target :: BB.IK.Solver.target()
  @type positions :: BB.IK.Solver.positions()
  @type meta :: BB.IK.Solver.meta()
  @type robot_or_context :: module() | Context.t()
  @type targets :: %{atom() => target()}

  @type motion_result :: {:ok, meta()} | {:error, atom(), meta()}
  @type solve_result :: {:ok, positions(), meta()} | {:error, atom(), meta()}
  @type multi_motion_result :: Motion.multi_motion_result()
  @type multi_solve_result :: Motion.multi_solve_result()

  @default_opts [
    max_iterations: 100,
    tolerance: 1.0e-4,
    lambda: 0.5,
    respect_limits: true
  ]

  @doc """
  Move an end-effector to a target position using DLS.

  This is a convenience wrapper around `BB.Motion.move_to/4` with the
  DLS solver pre-configured.

  ## Options

  DLS-specific:
  - `:max_iterations` - Maximum DLS iterations (default: 100)
  - `:tolerance` - Convergence tolerance in metres (default: 1.0e-4)
  - `:orientation_tolerance` - Orientation tolerance in radians (default: 0.01)
  - `:lambda` - Damping factor (default: 0.5)
  - `:adaptive_damping` - Adapt lambda based on error (default: true)
  - `:step_size` - Max joint update per iteration (default: 0.1)
  - `:respect_limits` - Whether to clamp to joint limits (default: true)

  Motion:
  - `:delivery` - How to send actuator commands: `:pubsub` (default), `:direct`, or `:sync`

  ## Returns

  - `{:ok, meta}` - Successfully moved; meta contains solver info
  - `{:error, reason, meta}` - Failed to reach target

  ## Examples

      BB.IK.DLS.Motion.move_to(MyRobot, :gripper, {0.3, 0.2, 0.1})

      BB.IK.DLS.Motion.move_to(context, :gripper, target,
        delivery: :direct,
        max_iterations: 200,
        tolerance: 0.001
      )
  """
  @spec move_to(robot_or_context(), atom(), target(), keyword()) :: motion_result()
  def move_to(robot_or_context, target_link, target, opts \\ []) do
    motion_opts = build_motion_opts(opts)
    Motion.move_to(robot_or_context, target_link, target, motion_opts)
  end

  @doc """
  Solve DLS without moving the robot.

  Useful for validating targets are reachable before committing to motion,
  or for planning multi-step movements.

  ## Options

  - `:max_iterations` - Maximum DLS iterations (default: 100)
  - `:tolerance` - Convergence tolerance in metres (default: 1.0e-4)
  - `:orientation_tolerance` - Orientation tolerance in radians (default: 0.01)
  - `:lambda` - Damping factor (default: 0.5)
  - `:adaptive_damping` - Adapt lambda based on error (default: true)
  - `:step_size` - Max joint update per iteration (default: 0.1)
  - `:respect_limits` - Whether to clamp to joint limits (default: true)

  ## Returns

  - `{:ok, positions, meta}` - Successfully solved
  - `{:error, reason, meta}` - Failed to solve

  ## Examples

      case BB.IK.DLS.Motion.solve(MyRobot, :gripper, target) do
        {:ok, positions, %{reached: true}} ->
          IO.puts("Target reachable")
          IO.inspect(positions)

        {:ok, _positions, %{reached: false, residual: residual}} ->
          IO.puts("Close but not exact, residual: \#{residual}m")

        {:error, :no_solution, _meta} ->
          IO.puts("Failed to converge")
      end
  """
  @spec solve(robot_or_context(), atom(), target(), keyword()) :: solve_result()
  def solve(robot_or_context, target_link, target, opts \\ []) do
    motion_opts = build_motion_opts(opts)
    Motion.solve_only(robot_or_context, target_link, target, motion_opts)
  end

  @doc """
  Move multiple end-effectors to target positions simultaneously using DLS.

  Useful for coordinated motion like walking gaits. Each target is solved
  independently using DLS and all actuator commands are sent together.

  ## Options

  Same as `move_to/4`.

  ## Returns

  - `{:ok, results}` - All targets solved; results is a map of link → `{:ok, positions, meta}`
  - `{:error, failed_link, reason, results}` - A target failed

  ## Examples

      targets = %{
        left_foot: {0.1, 0.0, 0.0},
        right_foot: {-0.1, 0.0, 0.0}
      }

      case BB.IK.DLS.Motion.move_to_multi(MyRobot, targets) do
        {:ok, results} ->
          IO.puts("All limbs positioned")

        {:error, failed_link, reason, _results} ->
          IO.puts("Failed to reach \#{failed_link}: \#{reason}")
      end
  """
  @spec move_to_multi(robot_or_context(), targets(), keyword()) :: multi_motion_result()
  def move_to_multi(robot_or_context, targets, opts \\ []) do
    motion_opts = build_motion_opts(opts)
    Motion.move_to_multi(robot_or_context, targets, motion_opts)
  end

  @doc """
  Solve DLS for multiple targets without moving the robot.

  Useful for validating that all targets in a coordinated motion are reachable.

  ## Options

  Same as `solve/4`.

  ## Returns

  - `{:ok, results}` - All targets solved
  - `{:error, failed_link, reason, results}` - A target failed

  ## Examples

      targets = %{left_foot: {0.1, 0.0, 0.0}, right_foot: {-0.1, 0.0, 0.0}}

      case BB.IK.DLS.Motion.solve_multi(MyRobot, targets) do
        {:ok, results} ->
          Enum.each(results, fn {link, {:ok, _pos, meta}} ->
            IO.puts("\#{link}: \#{meta.residual}m residual")
          end)

        {:error, failed_link, reason, _results} ->
          IO.puts("\#{failed_link} unreachable: \#{reason}")
      end
  """
  @spec solve_multi(robot_or_context(), targets(), keyword()) :: multi_solve_result()
  def solve_multi(robot_or_context, targets, opts \\ []) do
    motion_opts = build_motion_opts(opts)
    Motion.solve_only_multi(robot_or_context, targets, motion_opts)
  end

  defp build_motion_opts(opts) do
    dls_opts =
      @default_opts
      |> Keyword.merge(
        Keyword.take(opts, [
          :max_iterations,
          :tolerance,
          :orientation_tolerance,
          :lambda,
          :adaptive_damping,
          :step_size,
          :respect_limits,
          :exclude_joints
        ])
      )

    opts
    |> Keyword.take([:delivery])
    |> Keyword.merge(dls_opts)
    |> Keyword.put(:solver, DLS)
  end
end
