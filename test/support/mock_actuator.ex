# SPDX-FileCopyrightText: 2026 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.TestRobots.MockActuator do
  @moduledoc """
  Minimal mock actuator for testing — a proper `BB.Actuator` callback module
  that accepts commands and discards them.
  """
  use BB.Actuator, options_schema: []

  @impl BB.Actuator
  def init(opts), do: {:ok, %{opts: opts}}

  @impl BB.Actuator
  def disarm(_opts), do: :ok

  @impl BB.Actuator
  def handle_command(_message, state), do: {:noreply, state}
end
