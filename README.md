<!--
SPDX-FileCopyrightText: 2025 James Harton

SPDX-License-Identifier: Apache-2.0
-->

<img src="https://github.com/beam-bots/bb/blob/main/logos/beam_bots_logo.png?raw=true" alt="Beam Bots Logo" width="250" />

# BB.IK.DLS

[![CI](https://github.com/beam-bots/bb_ik_dls/actions/workflows/ci.yml/badge.svg)](https://github.com/beam-bots/bb_ik_dls/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hex version badge](https://img.shields.io/hexpm/v/bb_ik_dls.svg)](https://hex.pm/packages/bb_ik_dls)
[![REUSE status](https://api.reuse.software/badge/github.com/beam-bots/bb_ik_dls)](https://api.reuse.software/info/github.com/beam-bots/bb_ik_dls)

A Damped Least Squares (Levenberg-Marquardt) inverse kinematics solver for the [Beam Bots](https://github.com/beam-bots/bb) robotics framework.

DLS computes the joint angles needed to position an end-effector at a target location and orientation. The damping factor prevents instability near kinematic singularities where standard pseudoinverse methods become ill-conditioned.

## Features

- **Position and orientation solving** - supports position-only, quaternion, and axis-direction constraints
- **Numerical Jacobian** - works with any kinematic chain without analytical derivation
- **Adaptive damping** - automatically adjusts damping factor based on error reduction
- **Joint limit clamping** - respects configured joint limits
- **Nx tensor computation** - efficient numerical operations via Nx
- **Best-effort results** - returns closest solution even on failure
- **Continuous tracking** - GenServer for real-time target following

## Installation

Add `bb_ik_dls` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:bb_ik_dls, "~> 0.2.0"}
  ]
end
```

## Usage

### Basic Solving

```elixir
robot = MyRobot.robot()
{:ok, state} = BB.Robot.State.new(robot)

# Solve for end-effector to reach target position
target = Vec3.new(0.4, 0.2, 0.1)

case BB.IK.DLS.solve(robot, state, :end_effector, target) do
  {:ok, positions, meta} ->
    BB.Robot.State.set_positions(state, positions)
    IO.puts("Solved in #{meta.iterations} iterations")

  {:error, %BB.Error.Kinematics.NoSolution{residual: residual}} ->
    IO.puts("Failed to converge, residual: #{residual}m")
end
```

### With Orientation

```elixir
# Full 6-DOF target via Transform
target = Transform.from_position_quaternion(
  Vec3.new(0.3, 0.2, 0.1),
  Quaternion.from_axis_angle(Vec3.unit_z(), :math.pi() / 4)
)

{:ok, positions, meta} = BB.IK.DLS.solve(robot, state, :gripper, target)

# Or with axis constraint (point tool in a direction)
target = {Vec3.new(0.3, 0.2, 0.1), {:axis, Vec3.unit_z()}}
```

### Motion Convenience API

```elixir
# Move end-effector using BB.Motion integration
case BB.IK.DLS.Motion.move_to(MyRobot, :gripper, {0.3, 0.2, 0.1}) do
  {:ok, meta} -> IO.puts("Reached in #{meta.iterations} iterations")
  {:error, reason, _meta} -> IO.puts("Failed: #{reason}")
end

# Coordinated multi-limb motion
targets = %{left_foot: {0.1, 0.0, 0.0}, right_foot: {-0.1, 0.0, 0.0}}
BB.IK.DLS.Motion.move_to_multi(MyRobot, targets)
```

### Continuous Tracking

```elixir
# Start tracking a moving target
{:ok, tracker} = BB.IK.DLS.Tracker.start_link(
  robot: MyRobot,
  target_link: :gripper,
  initial_target: {0.3, 0.2, 0.1},
  update_rate: 30
)

# Update target from vision callback
BB.IK.DLS.Tracker.update_target(tracker, {0.35, 0.25, 0.15})

# Stop tracking
{:ok, final_positions} = BB.IK.DLS.Tracker.stop(tracker)
```

## Target Formats

| Format | Description |
|--------|-------------|
| `Vec3.t()` | Position-only target |
| `Transform.t()` | Position + full orientation from transform |
| `{Vec3.t(), {:quaternion, Quaternion.t()}}` | Position + explicit quaternion |
| `{Vec3.t(), {:axis, Vec3.t()}}` | Position + tool axis direction constraint |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `:max_iterations` | 100 | Maximum solver iterations |
| `:tolerance` | 1.0e-4 | Position convergence tolerance (metres) |
| `:orientation_tolerance` | 0.01 | Orientation convergence (radians) |
| `:lambda` | 0.5 | Damping factor |
| `:adaptive_damping` | true | Adjust lambda based on error reduction |
| `:step_size` | 0.1 | Maximum joint update per iteration (radians) |
| `:respect_limits` | true | Clamp solved values to joint limits |
| `:exclude_joints` | [] | Joint names to exclude from solving (e.g. grippers) |

## Algorithm

DLS uses the damped pseudoinverse to compute joint updates:

```
Δθ = Jᵀ (J Jᵀ + λ²I)⁻¹ e
```

where:
- **J** is the Jacobian matrix relating joint velocities to end-effector velocity
- **e** is the pose error vector
- **λ** is the damping factor
- **Δθ** is the joint update

The damping factor prevents instability when J Jᵀ is near-singular (at kinematic singularities).

## Comparison with FABRIK

| Aspect | DLS | FABRIK |
|--------|-----|--------|
| Orientation control | Excellent | Limited |
| 6-DOF arms | Well suited | Struggles |
| Speed per iteration | Slower | Faster |
| Algorithm complexity | Matrix operations | Geometric |
| Singularity handling | Damping | None |

## Documentation

Full documentation is available at [HexDocs](https://hexdocs.pm/bb_ik_dls).
