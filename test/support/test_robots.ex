# SPDX-FileCopyrightText: 2025 James Harton
#
# SPDX-License-Identifier: Apache-2.0

defmodule BB.IK.TestRobots do
  @moduledoc """
  Test robot definitions for IK solver testing.
  """

  defmodule TwoLinkArm do
    @moduledoc """
    Simple 2-DOF planar arm for basic IK testing.

    Structure:
    - base_link
      - shoulder_joint (revolute, Z-axis)
        - link1 (0.3m long)
          - elbow_joint (revolute, Z-axis)
            - link2
              - tip_joint (fixed, 0.2m offset)
                - tip

    Total reach: 0.5m from origin
    """
    use BB
    import BB.Unit

    settings do
      name(:two_link_arm)
    end

    topology do
      link :base_link do
        joint :shoulder_joint do
          type(:revolute)

          axis do
          end

          limit do
            lower(~u(-180 degree))
            upper(~u(180 degree))
            effort(~u(10 newton_meter))
            velocity(~u(90 degree_per_second))
          end

          link :link1 do
            joint :elbow_joint do
              type(:revolute)

              origin do
                x(~u(0.3 meter))
              end

              axis do
              end

              limit do
                lower(~u(-135 degree))
                upper(~u(135 degree))
                effort(~u(5 newton_meter))
                velocity(~u(90 degree_per_second))
              end

              link :link2 do
                joint :tip_joint do
                  type(:fixed)

                  origin do
                    x(~u(0.2 meter))
                  end

                  link(:tip)
                end
              end
            end
          end
        end
      end
    end
  end

  defmodule ThreeLinkArm do
    @moduledoc """
    3-DOF arm with vertical reach for testing 3D IK.

    Structure:
    - base_link
      - joint1 (revolute, Z-axis) - base rotation
        - link1
          - joint2 (revolute, Y-axis) - shoulder lift
            - link2 (0.2m)
              - joint3 (revolute, Y-axis) - elbow
                - link3
                  - tip_joint (fixed, 0.15m)
                    - tip
    """
    use BB
    import BB.Unit

    settings do
      name(:three_link_arm)
    end

    topology do
      link :base_link do
        joint :joint1 do
          type(:revolute)

          origin do
            z(~u(0.05 meter))
          end

          axis do
          end

          limit do
            lower(~u(-180 degree))
            upper(~u(180 degree))
            effort(~u(10 newton_meter))
            velocity(~u(90 degree_per_second))
          end

          link :link1 do
            joint :joint2 do
              type(:revolute)

              axis do
                roll(~u(-90 degree))
              end

              limit do
                lower(~u(-90 degree))
                upper(~u(90 degree))
                effort(~u(10 newton_meter))
                velocity(~u(90 degree_per_second))
              end

              link :link2 do
                joint :joint3 do
                  type(:revolute)

                  origin do
                    z(~u(0.2 meter))
                  end

                  axis do
                    roll(~u(-90 degree))
                  end

                  limit do
                    lower(~u(-135 degree))
                    upper(~u(135 degree))
                    effort(~u(5 newton_meter))
                    velocity(~u(90 degree_per_second))
                  end

                  link :link3 do
                    joint :tip_joint do
                      type(:fixed)

                      origin do
                        z(~u(0.15 meter))
                      end

                      link(:tip)
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  defmodule FixedOnlyChain do
    @moduledoc """
    A chain with only fixed joints - should fail with :no_dofs.
    """
    use BB
    import BB.Unit

    settings do
      name(:fixed_only)
    end

    topology do
      link :base_link do
        joint :fixed_joint do
          type(:fixed)

          origin do
            z(~u(0.1 meter))
          end

          link(:end_link)
        end
      end
    end
  end

  defmodule PrismaticArm do
    @moduledoc """
    Arm with a prismatic (linear) joint for testing prismatic IK.

    Structure:
    - base_link
      - rotate_joint (revolute, Z-axis)
        - link1
          - slide_joint (prismatic, X-axis, 0-0.3m range)
            - link2
              - tip_joint (fixed)
                - tip

    The prismatic joint allows linear extension along X.
    """
    use BB
    import BB.Unit

    settings do
      name(:prismatic_arm)
    end

    topology do
      link :base_link do
        joint :rotate_joint do
          type(:revolute)

          axis do
          end

          limit do
            lower(~u(-180 degree))
            upper(~u(180 degree))
            effort(~u(10 newton_meter))
            velocity(~u(90 degree_per_second))
          end

          link :link1 do
            joint :slide_joint do
              type(:prismatic)

              origin do
                x(~u(0.2 meter))
              end

              axis do
                pitch(~u(90 degree))
              end

              limit do
                lower(~u(0 meter))
                upper(~u(0.3 meter))
                effort(~u(50 newton))
                velocity(~u(0.5 meter_per_second))
              end

              link :link2 do
                joint :tip_joint do
                  type(:fixed)

                  origin do
                    x(~u(0.1 meter))
                  end

                  link(:tip)
                end
              end
            end
          end
        end
      end
    end
  end

  defmodule ContinuousJointArm do
    @moduledoc """
    Arm with continuous (unlimited rotation) joints.

    Structure:
    - base_link
      - wheel_joint (continuous, Z-axis) - can rotate infinitely
        - link1 (0.3m)
          - tip_joint (fixed, 0.2m)
            - tip

    Continuous joints have no position limits.
    """
    use BB
    import BB.Unit

    settings do
      name(:continuous_arm)
    end

    topology do
      link :base_link do
        joint :wheel_joint do
          type(:continuous)

          axis do
          end

          link :link1 do
            joint :tip_joint do
              type(:fixed)

              origin do
                x(~u(0.3 meter))
              end

              link(:tip)
            end
          end
        end
      end
    end
  end

  defmodule SixDofArm do
    @moduledoc """
    6-DOF anthropomorphic arm for testing orientation-constrained IK.

    Structure:
    - base_link
      - shoulder_yaw (revolute, Z-axis)
        - shoulder_link1
          - shoulder_pitch (revolute, Y-axis)
            - shoulder_link2
              - shoulder_roll (revolute, X-axis)
                - upper_arm (0.25m)
                  - elbow_pitch (revolute, Y-axis)
                    - forearm (0.2m)
                      - wrist_pitch (revolute, Y-axis)
                        - wrist_link
                          - wrist_roll (revolute, X-axis)
                            - hand
                              - tip_joint (fixed, 0.1m)
                                - tip

    Total: 6 revolute joints providing full 6-DOF control
    - Shoulder: yaw (Z), pitch (Y), roll (X) - spherical wrist
    - Elbow: pitch (Y) - single axis bend
    - Wrist: pitch (Y), roll (X) - allows tool orientation

    This arm can achieve arbitrary position and orientation within its workspace.
    Total reach: ~0.55m from shoulder
    """
    use BB
    import BB.Unit

    settings do
      name(:six_dof_arm)
    end

    topology do
      link :base_link do
        # Shoulder yaw - rotation around vertical axis
        joint :shoulder_yaw do
          type(:revolute)

          axis do
          end

          limit do
            lower(~u(-180 degree))
            upper(~u(180 degree))
            effort(~u(20 newton_meter))
            velocity(~u(90 degree_per_second))
          end

          link :shoulder_link1 do
            # Shoulder pitch - forward/back rotation
            joint :shoulder_pitch do
              type(:revolute)

              axis do
                roll(~u(-90 degree))
              end

              limit do
                lower(~u(-90 degree))
                upper(~u(90 degree))
                effort(~u(20 newton_meter))
                velocity(~u(90 degree_per_second))
              end

              link :shoulder_link2 do
                # Shoulder roll - rotation around arm axis
                joint :shoulder_roll do
                  type(:revolute)

                  axis do
                    pitch(~u(90 degree))
                  end

                  limit do
                    lower(~u(-180 degree))
                    upper(~u(180 degree))
                    effort(~u(10 newton_meter))
                    velocity(~u(120 degree_per_second))
                  end

                  link :upper_arm do
                    # Elbow pitch
                    joint :elbow_pitch do
                      type(:revolute)

                      origin do
                        z(~u(0.25 meter))
                      end

                      axis do
                        roll(~u(-90 degree))
                      end

                      limit do
                        lower(~u(-135 degree))
                        upper(~u(135 degree))
                        effort(~u(10 newton_meter))
                        velocity(~u(90 degree_per_second))
                      end

                      link :forearm do
                        # Wrist pitch
                        joint :wrist_pitch do
                          type(:revolute)

                          origin do
                            z(~u(0.2 meter))
                          end

                          axis do
                            roll(~u(-90 degree))
                          end

                          limit do
                            lower(~u(-90 degree))
                            upper(~u(90 degree))
                            effort(~u(5 newton_meter))
                            velocity(~u(120 degree_per_second))
                          end

                          link :wrist_link do
                            # Wrist roll
                            joint :wrist_roll do
                              type(:revolute)

                              axis do
                                pitch(~u(90 degree))
                              end

                              limit do
                                lower(~u(-180 degree))
                                upper(~u(180 degree))
                                effort(~u(2 newton_meter))
                                velocity(~u(180 degree_per_second))
                              end

                              link :hand do
                                joint :tip_joint do
                                  type(:fixed)

                                  origin do
                                    z(~u(0.1 meter))
                                  end

                                  link(:tip)
                                end
                              end
                            end
                          end
                        end
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  defmodule CollisionArm do
    @moduledoc """
    2-DOF arm with collision geometry for testing collision-aware IK.

    Structure:
    - base_link (with collision sphere at origin)
      - shoulder (revolute, Y-axis) - lifts arm in XZ plane
        - upper_arm (0.3m capsule along X)
          - elbow (revolute, Y-axis) - bends forearm
            - forearm (0.25m capsule along X)
              - tip_joint (fixed)
                - tip

    When the arm folds back (shoulder back + elbow bent), forearm can collide with base.
    Total reach: 0.55m from shoulder along X axis.
    """
    use BB
    import BB.Unit

    settings do
      name(:collision_arm)
    end

    topology do
      link :base_link do
        collision do
          sphere do
            radius(~u(0.08 meter))
          end
        end

        joint :shoulder do
          type(:revolute)

          axis do
            roll(~u(-90 degree))
          end

          limit do
            lower(~u(-135 degree))
            upper(~u(135 degree))
            effort(~u(10 newton_meter))
            velocity(~u(90 degree_per_second))
          end

          link :upper_arm do
            collision do
              origin do
                x(~u(0.15 meter))
              end

              capsule do
                radius(~u(0.03 meter))
                height(~u(0.3 meter))
              end
            end

            joint :elbow do
              type(:revolute)

              origin do
                x(~u(0.3 meter))
              end

              axis do
                roll(~u(-90 degree))
              end

              limit do
                lower(~u(-150 degree))
                upper(~u(150 degree))
                effort(~u(5 newton_meter))
                velocity(~u(90 degree_per_second))
              end

              link :forearm do
                collision do
                  origin do
                    x(~u(0.125 meter))
                  end

                  capsule do
                    radius(~u(0.025 meter))
                    height(~u(0.25 meter))
                  end
                end

                joint :tip_joint do
                  type(:fixed)

                  origin do
                    x(~u(0.25 meter))
                  end

                  link(:tip)
                end
              end
            end
          end
        end
      end
    end
  end
end
