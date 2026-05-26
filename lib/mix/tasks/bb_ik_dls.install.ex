# SPDX-FileCopyrightText: 2026 James Harton
#
# SPDX-License-Identifier: Apache-2.0

if Code.ensure_loaded?(Igniter) do
  defmodule Mix.Tasks.BbIkDls.Install do
    @shortdoc "Installs BB.IK.DLS into a project"
    @moduledoc """
    #{@shortdoc}

    The DLS solver is passed per-call via the `:solver` option to
    `BB.Motion` functions or `BB.Command.MoveTo` entries, so this
    installer prints a usage snippet rather than editing the topology.

    ## Example

    ```bash
    mix igniter.install bb_ik_dls
    ```
    """

    use Igniter.Mix.Task

    @impl Igniter.Mix.Task
    def info(_argv, _parent) do
      %Igniter.Mix.Task.Info{}
    end

    @impl Igniter.Mix.Task
    def igniter(igniter) do
      Igniter.add_notice(igniter, notice())
    end

    defp notice do
      """
      bb_ik_dls: pass `solver: BB.IK.DLS` to BB.Motion calls or MoveTo commands.

      Ad-hoc:

          BB.Motion.move_to(MyRobot, :gripper, target, solver: BB.IK.DLS)

      In the DSL:

          command :go_to, BB.Command.MoveTo,
            link: :gripper,
            solver: BB.IK.DLS

      For continuous target tracking, see BB.IK.DLS.Tracker — start_link/1
      into your supervision tree with a target link and update rate.
      """
    end
  end
else
  defmodule Mix.Tasks.BbIkDls.Install do
    @shortdoc "Installs BB.IK.DLS into a project"
    @moduledoc false
    use Mix.Task

    def run(_argv) do
      Mix.shell().error("""
      The bb_ik_dls.install task requires igniter. Please install igniter and try again.

          mix igniter.install bb_ik_dls
      """)

      exit({:shutdown, 1})
    end
  end
end
