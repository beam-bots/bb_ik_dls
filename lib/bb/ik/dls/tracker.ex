# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.DLS.Tracker do
  @moduledoc """
  GenServer implementing continuous position tracking with DLS.

  Runs a periodic IK solve loop, continuously solving for updated targets
  and sending actuator commands. Useful for following moving targets or
  real-time position control from external sources.

  ## Usage

      # Start tracking
      {:ok, pid} = BB.IK.DLS.Tracker.start_link(
        robot: MyRobot,
        target_link: :gripper,
        initial_target: {0.3, 0.2, 0.1},
        update_rate: 30
      )

      # Update target from vision callback
      BB.IK.DLS.Tracker.update_target(pid, {0.35, 0.25, 0.15})

      # Check status
      %{residual: 0.001, tracking: true} = BB.IK.DLS.Tracker.status(pid)

      # Stop and get final positions
      {:ok, positions} = BB.IK.DLS.Tracker.stop(pid)

  ## Options

  - `:robot` - Robot module (required)
  - `:target_link` - Link to track (required)
  - `:initial_target` - Starting target position (required)
  - `:update_rate` - Solve frequency in Hz (default: 20)
  - `:delivery` - Actuator command delivery: `:direct` (default), `:pubsub`, `:sync`
  - `:max_iterations` - Maximum DLS iterations per update (default: 100)
  - `:tolerance` - Convergence tolerance in metres (default: 1.0e-4)
  - `:lambda` - Damping factor (default: 0.5)
  - `:adaptive_damping` - Adapt lambda based on error (default: true)
  - `:step_size` - Max joint update per iteration (default: 0.1)
  - `:respect_limits` - Whether to clamp to joint limits (default: true)
  - `:name` - Optional GenServer name for registration

  ## Notes

  - Uses `:direct` delivery by default for low latency
  - Continues tracking even if individual solves fail (best-effort)
  - Call `stop/1` to cleanly terminate tracking
  """

  use GenServer

  alias BB.IK.DLS
  alias BB.Motion
  alias BB.Robot.Runtime
  alias BB.Robot.State, as: RobotState

  defstruct [
    :robot_module,
    :robot,
    :robot_state,
    :target_link,
    :target,
    :delivery,
    :solver_opts,
    :update_rate,
    :timer_ref,
    :last_positions,
    :last_meta,
    :last_update,
    tracking: true
  ]

  @default_update_rate 20
  @default_delivery :direct

  @doc """
  Start a tracker process.

  See module documentation for options.
  """
  def start_link(opts) do
    {gen_opts, tracker_opts} = Keyword.split(opts, [:name])
    GenServer.start_link(__MODULE__, tracker_opts, gen_opts)
  end

  @doc """
  Update the current target position.
  """
  def update_target(tracker, target) do
    GenServer.call(tracker, {:update_target, target})
  end

  @doc """
  Get current tracking status.
  """
  def status(tracker) do
    GenServer.call(tracker, :status)
  end

  @doc """
  Stop tracking and return final positions.

  ## Options

  - `:hold` - Send hold commands to actuators (default: false)
  """
  def stop(tracker, opts \\ []) do
    GenServer.call(tracker, {:stop, opts})
  end

  @impl GenServer
  def init(opts) do
    robot_module = Keyword.fetch!(opts, :robot)
    target_link = Keyword.fetch!(opts, :target_link)
    initial_target = Keyword.fetch!(opts, :initial_target)

    update_rate = Keyword.get(opts, :update_rate, @default_update_rate)
    delivery = Keyword.get(opts, :delivery, @default_delivery)

    solver_opts =
      Keyword.take(opts, [
        :max_iterations,
        :tolerance,
        :orientation_tolerance,
        :lambda,
        :adaptive_damping,
        :step_size,
        :respect_limits
      ])
      |> Keyword.reject(fn {_k, v} -> is_nil(v) end)

    robot = Runtime.get_robot(robot_module)
    robot_state = Runtime.get_robot_state(robot_module)

    state = %__MODULE__{
      robot_module: robot_module,
      robot: robot,
      robot_state: robot_state,
      target_link: target_link,
      target: initial_target,
      delivery: delivery,
      solver_opts: solver_opts,
      update_rate: update_rate,
      tracking: true
    }

    timer_ref = schedule_tick(update_rate)

    {:ok, %{state | timer_ref: timer_ref}}
  end

  @impl GenServer
  def handle_call({:update_target, target}, _from, state) do
    if state.tracking do
      {:reply, {:ok, state}, %{state | target: target}}
    else
      {:reply, {:error, :not_tracking}, state}
    end
  end

  @impl GenServer
  def handle_call(:status, _from, state) do
    status = %{
      tracking: state.tracking,
      target: state.target,
      residual: get_in(state.last_meta, [:residual]),
      iterations: get_in(state.last_meta, [:iterations]) || 0,
      update_rate: state.update_rate,
      last_update: state.last_update
    }

    {:reply, status, state}
  end

  @impl GenServer
  def handle_call({:stop, opts}, _from, state) do
    if state.timer_ref, do: Process.cancel_timer(state.timer_ref)

    if opts[:hold] do
      send_hold_commands(state)
    end

    {:stop, :normal, {:ok, state.last_positions || %{}}, %{state | tracking: false}}
  end

  @impl GenServer
  def handle_info(:tick, state) do
    state = do_solve_and_send(state)

    timer_ref =
      if state.tracking do
        schedule_tick(state.update_rate)
      else
        nil
      end

    {:noreply, %{state | timer_ref: timer_ref}}
  end

  defp do_solve_and_send(state) do
    motion_opts =
      state.solver_opts
      |> Keyword.put(:solver, DLS)
      |> Keyword.put(:delivery, state.delivery)

    case Motion.move_to(state.robot_module, state.target_link, state.target, motion_opts) do
      {:ok, meta} ->
        positions = get_current_positions(state)

        %{
          state
          | last_positions: positions,
            last_meta: meta,
            last_update: DateTime.utc_now()
        }

      {:error, error} ->
        meta = %{
          residual: Map.get(error, :residual),
          iterations: Map.get(error, :iterations, 0)
        }

        %{state | last_meta: meta}
    end
  end

  defp get_current_positions(state) do
    case Map.get(state.robot.joints, state.target_link) do
      nil ->
        %{}

      _joint ->
        state.robot.joints
        |> Map.keys()
        |> Enum.reduce(%{}, fn joint_name, acc ->
          pos = RobotState.get_joint_position(state.robot_state, joint_name)
          Map.put(acc, joint_name, pos)
        end)
    end
  end

  defp send_hold_commands(state) do
    Enum.each(state.robot.actuators, fn {name, _info} ->
      BB.Actuator.hold!(state.robot_module, name)
    end)
  end

  defp schedule_tick(update_rate) do
    interval = div(1000, update_rate)
    Process.send_after(self(), :tick, interval)
  end
end
