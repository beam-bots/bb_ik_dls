<!--
SPDX-FileCopyrightText: 2025 James Harton

SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

BB.IK.DLS is a Damped Least Squares (Levenberg-Marquardt) inverse kinematics solver for the Beam Bots robotics framework. It computes joint angles needed to position an end-effector at a target location and orientation.

The solver implements the `BB.IK.Solver` behaviour from the core BB framework, making it interchangeable with other IK algorithms like FABRIK.

## Architecture

```
lib/bb/ik/
├── dls.ex           # Main module, BB.IK.Solver implementation
└── dls/
    ├── algorithm.ex # Core DLS iteration loop with Nx.Defn
    ├── jacobian.ex  # Numerical Jacobian computation
    ├── motion.ex    # BB.Motion convenience wrappers
    └── tracker.ex   # GenServer for continuous tracking
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `BB.IK.DLS` | Main entry point. Implements `BB.IK.Solver` behaviour. Provides `solve/5` and `solve_and_update/5` |
| `BB.IK.DLS.Algorithm` | Core iteration using damped pseudoinverse. Contains `Nx.Defn` for performance |
| `BB.IK.DLS.Jacobian` | Computes numerical Jacobian via finite differences. Works with any kinematic chain |
| `BB.IK.DLS.Motion` | Wraps `BB.Motion` with DLS pre-configured. Convenience API for `move_to`, `solve`, and multi-target operations |
| `BB.IK.DLS.Tracker` | GenServer for continuous position tracking at configurable update rates |

## Build and Test Commands

```bash
mix check --no-retry    # Run all checks (compile, test, format, credo, dialyzer, reuse)
mix test                # Run tests
mix test path/to/test.exs:42  # Run single test at line
mix format              # Format code
mix credo --strict      # Linting
```

The project uses `ex_check` - always prefer `mix check --no-retry` over running individual tools.

## Key Patterns

### Target Formats

The solver accepts multiple target formats:
- `Vec3.t()` - position only
- `Transform.t()` - position + orientation from 4x4 matrix
- `{Vec3.t(), {:quaternion, Quaternion.t()}}` - position + explicit quaternion
- `{Vec3.t(), {:axis, Vec3.t()}}` - position + tool axis direction

### Error Handling

Solver returns structured errors from `BB.Error.Kinematics`:
- `UnknownLink` - target link not in topology
- `NoDofs` - chain has no movable joints
- `NoSolution` - failed to converge (includes best-effort positions)

### Nx Usage

The `Algorithm` module uses `Nx.Defn` for the core computation:
```elixir
defn compute_update(jacobian, error, lambda) do
  jjt = Nx.dot(jacobian, Nx.transpose(jacobian))
  # ...
end
```

All tensor operations should use Nx, following the BB ecosystem convention.

## Test Structure

Tests use robot fixtures from `test/support/test_robots.ex` (compiled via `elixirc_paths`):
- `TwoLinkArm` - 2-DOF planar arm (0.5m reach)
- `ThreeLinkArm` - 3-DOF arm with vertical reach
- `SixDofArm` - 6-DOF anthropomorphic arm for orientation tests
- `PrismaticArm` - mixed revolute/prismatic joints
- `ContinuousJointArm` - unlimited rotation joints
- `FixedOnlyChain` - no movable joints (error case)

## Dependencies

- `bb ~> 0.10` - Core framework (provides `BB.IK.Solver` behaviour, `BB.Robot`, `BB.Math.*`)
- `nx` - Numerical computing (via bb)

## When Making Changes

1. Run `mix check --no-retry` after any changes
2. Jacobian computation uses finite differences with ε = 1.0e-6
3. Adaptive damping adjusts λ by ×0.9 on error reduction, ×1.5 on increase
4. Lambda is clamped to [1.0e-6, 100.0]
5. The `Tracker` GenServer uses `:direct` delivery by default for low latency
