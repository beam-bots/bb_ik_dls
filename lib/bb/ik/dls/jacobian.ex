# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS.Jacobian do
  @moduledoc """
  Numerical Jacobian computation for inverse kinematics.

  The Jacobian matrix relates joint velocities to end-effector velocities:

      ẋ = J(θ) θ̇

  where ẋ is the end-effector velocity (6D for position + orientation),
  θ̇ is the joint velocity vector, and J is the Jacobian matrix.

  This module computes the Jacobian numerically using finite differences,
  which works with any kinematic chain without requiring analytical derivation.
  """

  alias BB.Math.Quaternion
  alias BB.Math.Transform
  alias BB.Math.Vec3
  alias BB.Robot
  alias BB.Robot.Kinematics

  @epsilon 1.0e-6

  @doc """
  Compute the position-only Jacobian (3xN matrix).

  Returns a 3xN Nx tensor where N is the number of joints.
  Each column j represents the change in end-effector position
  per unit change in joint j.

  ## Parameters

  - `robot` - The BB.Robot struct
  - `positions` - Current joint positions map `%{atom() => float()}`
  - `target_link` - The end-effector link name
  - `joint_names` - Ordered list of joint names in the kinematic chain

  ## Returns

  An Nx tensor of shape `{3, n}` where n is the length of joint_names.
  """
  @spec compute(Robot.t(), map(), atom(), [atom()]) :: Nx.Tensor.t()
  def compute(robot, positions, target_link, joint_names) do
    columns =
      Enum.map(joint_names, fn joint_name ->
        compute_position_column(robot, positions, target_link, joint_name)
      end)

    columns
    |> Nx.stack()
    |> Nx.transpose()
  end

  @doc """
  Compute the full Jacobian including orientation (6xN matrix).

  Returns a 6xN Nx tensor where N is the number of joints.
  Rows 0-2 represent position change, rows 3-5 represent orientation change
  (as angular velocity / axis-angle rate).

  ## Parameters

  Same as `compute/4`.

  ## Returns

  An Nx tensor of shape `{6, n}` where n is the length of joint_names.
  """
  @spec compute_with_orientation(Robot.t(), map(), atom(), [atom()]) :: Nx.Tensor.t()
  def compute_with_orientation(robot, positions, target_link, joint_names) do
    columns =
      Enum.map(joint_names, fn joint_name ->
        compute_full_column(robot, positions, target_link, joint_name)
      end)

    columns
    |> Nx.stack()
    |> Nx.transpose()
  end

  defp compute_position_column(robot, positions, target_link, joint_name) do
    current_pos = Map.get(positions, joint_name, 0.0)

    positions_plus = Map.put(positions, joint_name, current_pos + @epsilon)
    positions_minus = Map.put(positions, joint_name, current_pos - @epsilon)

    {x_plus, y_plus, z_plus} = Kinematics.link_position(robot, positions_plus, target_link)
    {x_minus, y_minus, z_minus} = Kinematics.link_position(robot, positions_minus, target_link)

    dx = (x_plus - x_minus) / (2 * @epsilon)
    dy = (y_plus - y_minus) / (2 * @epsilon)
    dz = (z_plus - z_minus) / (2 * @epsilon)

    Nx.tensor([dx, dy, dz], type: :f64)
  end

  defp compute_full_column(robot, positions, target_link, joint_name) do
    current_pos = Map.get(positions, joint_name, 0.0)

    positions_plus = Map.put(positions, joint_name, current_pos + @epsilon)
    positions_minus = Map.put(positions, joint_name, current_pos - @epsilon)

    transform_plus = Kinematics.forward_kinematics(robot, positions_plus, target_link)
    transform_minus = Kinematics.forward_kinematics(robot, positions_minus, target_link)

    pos_plus = Transform.get_translation(transform_plus)
    pos_minus = Transform.get_translation(transform_minus)

    dx = (Vec3.x(pos_plus) - Vec3.x(pos_minus)) / (2 * @epsilon)
    dy = (Vec3.y(pos_plus) - Vec3.y(pos_minus)) / (2 * @epsilon)
    dz = (Vec3.z(pos_plus) - Vec3.z(pos_minus)) / (2 * @epsilon)

    quat_plus = Transform.get_quaternion(transform_plus)
    quat_minus = Transform.get_quaternion(transform_minus)

    {d_omega_x, d_omega_y, d_omega_z} =
      compute_orientation_derivative(quat_minus, quat_plus, 2 * @epsilon)

    Nx.tensor([dx, dy, dz, d_omega_x, d_omega_y, d_omega_z], type: :f64)
  end

  defp compute_orientation_derivative(quat_minus, quat_plus, delta) do
    q_diff = Quaternion.multiply(quat_plus, Quaternion.conjugate(quat_minus))

    {axis, angle} = Quaternion.to_axis_angle(q_diff)

    omega_x = Vec3.x(axis) * angle / delta
    omega_y = Vec3.y(axis) * angle / delta
    omega_z = Vec3.z(axis) * angle / delta

    {omega_x, omega_y, omega_z}
  end
end
