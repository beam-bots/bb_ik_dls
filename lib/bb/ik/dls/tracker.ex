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
        source_link: :base_link,
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
  - `:source_link` - Link the chain starts at (required, no default)
  - `:initial_target` - Starting target position (required)
  - `:update_rate` - Solve frequency in Hz (default: 20)
  - `:delivery` - Actuator command delivery. `:direct` (default) casts each
    command and waits for nothing; `:pubsub` publishes it and waits for the
    actuator to accept it, which blocks the loop for as long as that takes
  - `:timeout` - How long to wait for each actuator under `:pubsub`, in
    milliseconds (default 5000). Ignored under `:direct`
  - `:max_iterations` - Maximum DLS iterations per update (default: 100)
  - `:tolerance` - Convergence tolerance in metres (default: 1.0e-4)
  - `:lambda` - Damping factor (default: 0.5)
  - `:adaptive_damping` - Adapt lambda based on error (default: true)
  - `:step_size` - Max joint update per iteration (default: 0.1)
  - `:respect_limits` - Whether to clamp to joint limits (default: true)
  - `:name` - Optional GenServer name for registration

  ## Position feedback is a prerequisite

  Every solve starts from the robot's current configuration, which is written
  from `BB.Message.Sensor.JointState` messages and from nothing else - a
  commanded position is not a measured one. A joint that nothing reports on
  therefore stays at its initial configuration, and the tracker re-solves from
  that same frozen pose on every tick.

  That does not diverge: a solve is a function of its seed and its target, so a
  frozen seed still yields an absolute joint configuration that reaches the
  target. What it loses is the warm start, and with it the continuity between
  ticks. Each solve pays the iteration count a distant seed needs rather than
  the handful a nearby one does, and converges less reliably near singularities
  - and because `:step_size` caps how far the configuration moves per iteration,
  `:max_iterations` bounds how far the answer can travel from its seed at all.
  The one that bites is that the answer stops depending on the path taken to
  reach the target, which leaves the arm free to change solution branch from one
  tick to the next: a redundant arm can be asked to swing between two equally
  valid postures inside a single tick period.

  So the tracked joints want something that reports where they are: an encoder,
  a driver that declares `:position_feedback` through
  `c:BB.Actuator.capabilities/1`, or `BB.Sensor.OpenLoopPositionEstimator`
  interpolating from the actuator's own `BeginMotion` messages. `BB.Dsl` warns
  at compile time about a driven joint with none of the three, and simulation
  supplies an estimator itself.

  ## Notes

  - Uses `:direct` delivery by default for low latency. Under `:pubsub` a solve
    that outlives `:timeout` exits the tracker, as `GenServer.call/3` does
  - Continues tracking even if individual solves fail (best-effort)
  - Call `stop/1` to cleanly terminate tracking
  """

  use GenServer

  require Logger

  alias BB.IK.DLS
  alias BB.Motion
  alias BB.Robot.Runtime

  defstruct [
    :robot_module,
    :robot,
    :target_link,
    :source_link,
    :target,
    :delivery,
    :timeout,
    :solver_opts,
    :update_rate,
    :loop,
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
  Stop tracking and return the configuration the last solve arrived at.

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
    source_link = Keyword.fetch!(opts, :source_link)
    initial_target = Keyword.fetch!(opts, :initial_target)

    update_rate = Keyword.get(opts, :update_rate, @default_update_rate)
    delivery = Keyword.get(opts, :delivery, @default_delivery)
    timeout = Keyword.get(opts, :timeout)

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

    state = %__MODULE__{
      robot_module: robot_module,
      robot: robot,
      target_link: target_link,
      source_link: source_link,
      target: initial_target,
      delivery: delivery,
      timeout: timeout,
      solver_opts: solver_opts,
      update_rate: update_rate,
      loop:
        BB.Loop.new(%{robot: robot_module, path: [:ik_tracker, target_link]},
          clock: {:rate, update_rate}
        ),
      tracking: true
    }

    {:ok, %{state | loop: BB.Loop.arm(state.loop)}}
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
    BB.Loop.cancel(state.loop)

    if opts[:hold] do
      send_hold_commands(state)
    end

    {:stop, :normal, {:ok, state.last_positions || %{}}, %{state | tracking: false}}
  end

  @impl GenServer
  def handle_info(:tick, state) do
    {_dt, _skipped, loop} = BB.Loop.tick(state.loop)
    state = do_solve_and_send(%{state | loop: loop})

    # tick/1 has already armed the next one, so stopping means cancelling it.
    loop = if state.tracking, do: state.loop, else: BB.Loop.cancel(state.loop)

    {:noreply, %{state | loop: loop}}
  end

  # Solving and sending separately rather than through `BB.Motion.move_to/4`,
  # because the tracker has to report the configuration it solved for and
  # `move_to/4` keeps that to itself.
  defp do_solve_and_send(state) do
    solve_opts =
      state.solver_opts
      |> Keyword.put(:solver, DLS)
      |> Keyword.put(:source_link, state.source_link)

    case Motion.solve_only(state.robot_module, state.target_link, state.target, solve_opts) do
      {:ok, positions, meta} ->
        send_positions(state, positions)

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

  defp send_positions(state, positions) do
    opts =
      [delivery: state.delivery, timeout: state.timeout]
      |> Keyword.reject(fn {_key, value} -> is_nil(value) end)

    case Motion.send_positions(state.robot_module, positions, opts) do
      :ok ->
        :ok

      {:error, error} ->
        Logger.warning(
          "Tracking #{inspect(state.target_link)} on #{inspect(state.robot_module)}: " <>
            "actuator refused its position command: #{Exception.message(error)}"
        )
    end
  end

  defp send_hold_commands(state) do
    Enum.each(state.robot.actuators, fn {name, _info} ->
      BB.Actuator.hold!(state.robot_module, name)
    end)
  end
end
