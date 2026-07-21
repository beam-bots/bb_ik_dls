<!--
SPDX-FileCopyrightText: 2026 James Harton

SPDX-License-Identifier: Apache-2.0
-->

# BB.IK.DLS Usage Rules

`bb_ik_dls` provides `BB.IK.DLS`, a Damped Least Squares (Levenberg-Marquardt)
implementation of the `BB.IK.Solver` behaviour for [Beam Bots](https://hexdocs.pm/bb).
For BB framework basics, see `bb`'s rules (`mix usage_rules.sync <file> bb:all`);
this file covers only what's specific to the solver.

## Core principles

1. **A solver is a module you pass, not a component you declare.** `BB.IK.DLS`
   is not wired into the `topology` and is not supervised — hand it to a motion
   call via the `:solver` option (or use the pre-configured `BB.IK.DLS.Motion`
   wrapper).
2. **DLS handles singularities.** The damping factor (`:lambda`) keeps the
   solve stable near kinematic singularities where a plain pseudoinverse blows
   up. That is the reason to choose DLS over FABRIK.
3. **Solver options are an untyped keyword list with DLS-specific defaults.**
   There is no schema, and defaults differ from other solvers — don't assume a
   value carries over from `bb_ik_fabrik`.

## Using it

Ad-hoc, through `BB.Motion`:

```elixir
BB.Motion.move_to(MyRobot.Robot, :gripper, {0.4, 0.2, 0.1}, solver: BB.IK.DLS)
```

Declaratively, as a `BB.Command.MoveTo` entry in the DSL:

```elixir
command :reach, BB.Command.MoveTo,
  link: :gripper,
  solver: BB.IK.DLS
```

Or via the convenience wrapper, which is `BB.Motion` with the solver pre-set:

```elixir
BB.IK.DLS.Motion.move_to(MyRobot.Robot, :gripper, {0.4, 0.2, 0.1})
```

Targets are `{x, y, z}` / `BB.Math.Vec3.t()` for position only, `{vec3,
orientation}` to constrain orientation, or a `BB.Math.Transform.t()`.

## Options

Passed through the motion call; all optional.

| Option | Default | Meaning |
|---|---|---|
| `:max_iterations` | `100` | Iteration cap (note: `BB.IK.FABRIK` defaults to `50`) |
| `:tolerance` | `1.0e-4` | Position convergence, metres |
| `:orientation_tolerance` | `0.01` | Orientation convergence, radians |
| `:lambda` | `0.5` | Damping factor |
| `:adaptive_damping` | `true` | Adjust `:lambda` from error reduction |

## Solving directly

For reachability checks without moving the robot, call the behaviour function.
It takes the compiled `%BB.Robot{}` struct (from `MyRobot.Robot.robot()`) and a
state or positions map — not the robot module:

```elixir
robot = MyRobot.Robot.robot()
{:ok, state} = BB.Robot.State.new(robot)

case BB.IK.DLS.solve(robot, state, :gripper, {0.4, 0.2, 0.1}) do
  {:ok, positions, meta} -> {positions, meta.iterations}
  {:error, %BB.Error.Kinematics.NoSolution{residual: r}} -> {:unreachable, r}
end
```

## Continuous tracking

To follow a moving target, run `BB.IK.DLS.Tracker` (a GenServer) in your
supervision tree; push new targets with `update_target/2`:

```elixir
{:ok, pid} = BB.IK.DLS.Tracker.start_link(
  robot: MyRobot.Robot, target_link: :gripper, initial_target: {0.3, 0.2, 0.1}, update_rate: 30
)
BB.IK.DLS.Tracker.update_target(pid, {0.35, 0.25, 0.15})
```

## Anti-patterns

- **Don't declare the solver in `topology`.** It is not a `BB.Sensor`/
  `BB.Actuator`/`BB.Controller` — there is no process to supervise. Pass
  `solver: BB.IK.DLS` per call.
- **Don't carry FABRIK's defaults over.** DLS tunes stability with `:lambda`
  and `:adaptive_damping`; its `:max_iterations` default is `100`, not `50`.
- **Don't pass the robot module to `solve/5`.** It wants the `%BB.Robot{}`
  struct plus a state/positions map.

## Further reading

- [bb_ik_dls docs](https://hexdocs.pm/bb_ik_dls)
- `bb`'s kinematics rules (`bb:kinematics`) and
  [Inverse Kinematics](https://hexdocs.pm/bb/09-inverse-kinematics.html)
