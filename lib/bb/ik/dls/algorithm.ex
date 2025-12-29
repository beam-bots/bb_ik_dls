# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS.Algorithm do
  @moduledoc """
  Core Damped Least Squares iteration algorithm.

  Implements the Levenberg-Marquardt style damped pseudoinverse:

      Δθ = J^T (J J^T + λ²I)^(-1) e

  where:
  - J is the Jacobian matrix
  - e is the pose error vector
  - λ is the damping factor
  - Δθ is the joint update

  The damping factor prevents instability near kinematic singularities
  where J J^T becomes ill-conditioned.
  """

  import Nx.Defn

  alias BB.IK.DLS.Jacobian
  alias BB.Math.Quaternion
  alias BB.Math.Transform
  alias BB.Math.Vec3
  alias BB.Robot.Kinematics

  @min_lambda 1.0e-6
  @max_lambda 100.0

  defmodule State do
    @moduledoc false
    defstruct [
      :robot,
      :target_link,
      :target_position,
      :target_orientation,
      :joint_names,
      :config,
      :positions,
      :lambda,
      :iteration,
      :prev_error_norm
    ]
  end

  @type config :: %{
          max_iterations: pos_integer(),
          tolerance: float(),
          orientation_tolerance: float(),
          lambda: float(),
          adaptive_damping: boolean(),
          step_size: float(),
          respect_limits: boolean()
        }

  @type iterate_result ::
          {:ok, map(), %{iterations: non_neg_integer(), converged: boolean()}}
          | {:error, :max_iterations, %{iterations: non_neg_integer(), positions: map()}}

  @doc """
  Run the DLS iteration loop until convergence or max iterations.

  ## Parameters

  - `robot` - The BB.Robot struct
  - `initial_positions` - Starting joint positions
  - `target_link` - End-effector link name
  - `target_position` - Target position as Vec3
  - `target_orientation` - Target orientation as Quaternion (or nil for position-only)
  - `joint_names` - Ordered list of joint names
  - `config` - Solver configuration map

  ## Returns

  - `{:ok, positions, meta}` - Converged successfully
  - `{:error, :max_iterations, meta}` - Failed to converge
  """
  @spec iterate(
          BB.Robot.t(),
          map(),
          atom(),
          Vec3.t(),
          Quaternion.t() | nil,
          [atom()],
          config()
        ) :: iterate_result()
  def iterate(
        robot,
        initial_positions,
        target_link,
        target_position,
        target_orientation,
        joint_names,
        config
      ) do
    state = %State{
      robot: robot,
      target_link: target_link,
      target_position: target_position,
      target_orientation: target_orientation,
      joint_names: joint_names,
      config: config,
      positions: initial_positions,
      lambda: config.lambda,
      iteration: 0,
      prev_error_norm: nil
    }

    do_iterate(state)
  end

  defp do_iterate(%State{iteration: iteration, config: config, positions: positions} = state) do
    if iteration >= config.max_iterations do
      {:error, :max_iterations, %{iterations: iteration, positions: positions}}
    else
      case check_convergence(state) do
        {:converged, iteration} ->
          {:ok, positions, %{iterations: iteration, converged: true}}

        {:continue, error_vector, error_norm} ->
          new_state = perform_iteration(state, error_vector, error_norm)
          do_iterate(new_state)
      end
    end
  end

  defp check_convergence(%State{} = state) do
    current_transform =
      Kinematics.forward_kinematics(state.robot, state.positions, state.target_link)

    current_position = Transform.get_translation(current_transform)
    position_error = compute_position_error(current_position, state.target_position)
    position_error_norm = Vec3.magnitude(position_error)

    {error_vector, orientation_error_norm} =
      build_error_vector(current_transform, position_error, state.target_orientation)

    converged =
      position_error_norm < state.config.tolerance and
        (is_nil(state.target_orientation) or
           orientation_error_norm < state.config.orientation_tolerance)

    if converged do
      {:converged, state.iteration}
    else
      error_norm = Nx.to_number(Nx.LinAlg.norm(error_vector))
      {:continue, error_vector, error_norm}
    end
  end

  defp build_error_vector(_current_transform, position_error, nil) do
    {Vec3.tensor(position_error), 0.0}
  end

  defp build_error_vector(current_transform, position_error, target_orientation) do
    current_orientation = Transform.get_quaternion(current_transform)
    orientation_error = compute_orientation_error(current_orientation, target_orientation)
    orientation_norm = Vec3.magnitude(orientation_error)

    error =
      Nx.concatenate([
        Vec3.tensor(position_error),
        Vec3.tensor(orientation_error)
      ])

    {error, orientation_norm}
  end

  defp perform_iteration(%State{} = state, error_vector, error_norm) do
    jacobian = compute_jacobian(state)
    delta_theta = compute_update(jacobian, error_vector, state.lambda)
    delta_theta = limit_step_size(delta_theta, state.config.step_size)
    new_positions = apply_update(state.positions, state.joint_names, delta_theta)

    new_lambda = update_lambda(state, error_norm)

    %{
      state
      | positions: new_positions,
        lambda: new_lambda,
        iteration: state.iteration + 1,
        prev_error_norm: error_norm
    }
  end

  defp compute_jacobian(%State{target_orientation: nil} = state) do
    Jacobian.compute(state.robot, state.positions, state.target_link, state.joint_names)
  end

  defp compute_jacobian(%State{} = state) do
    Jacobian.compute_with_orientation(
      state.robot,
      state.positions,
      state.target_link,
      state.joint_names
    )
  end

  defp update_lambda(%State{config: %{adaptive_damping: false}} = state, _error_norm) do
    state.lambda
  end

  defp update_lambda(%State{prev_error_norm: nil} = state, _error_norm) do
    state.lambda
  end

  defp update_lambda(%State{} = state, error_norm) do
    adapt_lambda(state.lambda, state.prev_error_norm, error_norm)
  end

  defp compute_position_error(current_position, target_position) do
    Vec3.subtract(target_position, current_position)
  end

  defp compute_orientation_error(current_orientation, target_orientation) do
    q_error = Quaternion.multiply(target_orientation, Quaternion.conjugate(current_orientation))
    {axis, angle} = Quaternion.to_axis_angle(q_error)
    Vec3.scale(axis, angle)
  end

  @doc """
  Compute the damped pseudoinverse update.

  Uses the formula: Δθ = J^T (J J^T + λ²I)^(-1) e

  This is a defn function for performance.
  """
  defn compute_update(jacobian, error, lambda) do
    jjt = Nx.dot(jacobian, Nx.transpose(jacobian))

    m = Nx.axis_size(jjt, 0)
    damping = Nx.multiply(lambda * lambda, Nx.eye(m, type: :f64))
    jjt_damped = Nx.add(jjt, damping)

    x = Nx.LinAlg.solve(jjt_damped, error)

    Nx.dot(Nx.transpose(jacobian), x)
  end

  defp limit_step_size(delta_theta, max_step) do
    norm = Nx.to_number(Nx.LinAlg.norm(delta_theta))

    if norm > max_step do
      scale = max_step / norm
      Nx.multiply(delta_theta, scale)
    else
      delta_theta
    end
  end

  defp apply_update(positions, joint_names, delta_theta) do
    delta_list = Nx.to_flat_list(delta_theta)

    joint_names
    |> Enum.zip(delta_list)
    |> Enum.reduce(positions, fn {joint_name, delta}, acc ->
      current = Map.get(acc, joint_name, 0.0)
      Map.put(acc, joint_name, current + delta)
    end)
  end

  defp adapt_lambda(lambda, prev_error, current_error) do
    new_lambda =
      if current_error < prev_error do
        lambda * 0.9
      else
        lambda * 1.5
      end

    new_lambda
    |> max(@min_lambda)
    |> min(@max_lambda)
  end
end
