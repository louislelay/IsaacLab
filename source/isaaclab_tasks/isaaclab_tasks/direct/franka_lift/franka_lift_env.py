# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

@configclass
class FrankaLiftEnvCfg(DirectRLEnvCfg):
    """Configuration class for the Franka lifting environment."""

    # Environment
    episode_length_s = 5.0
    decimation = 8
    action_space = 9
    observation_space = 21
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # Robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )


    # Markers
    markers = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Markers")
    markers.markers["frame"].scale = (0.1, 0.1, 0.1)

    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=markers,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
        ],
    )

    # Reset
    initial_joint_pos_range = [-0.1, 0.1]

    # Reward scales
    dist_reward_scale = 1.5
    action_penalty_scale = 0.05
    lift_reward_scale = 15
    height_reward_scale = 16


class FrankaLiftEnv(DirectRLEnv):
    """Environment class for the Franka robot lifting task."""

    cfg: FrankaLiftEnvCfg

    def __init__(self, cfg: FrankaLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.target_pos = torch.tensor([0.5, 0.0, 0.3], device=self.device).repeat(self.num_envs, 1)

    def _setup_scene(self) -> None:
        """Set up the simulation scene with the robot, table, and ground plane."""
        self._robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        # Add table
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        # self.scene.target_markers = VisualizationMarkers(self.cfg.markers)
        self.scene.ee_markers = VisualizationMarkers(self.cfg.markers)
        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        # Add articulation to the scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["ee_frame"] = self._ee_frame
        # Add light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Process actions before stepping the physics
        for the end-effector and target.
        """
        self.actions = actions.clone().clamp(self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        """Send position targets to the robot's joints."""
        self._robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        """
        Retrieve the robot's state for RL training:
          - Joint positions (scaled around defaults)
          - Joint velocities
          - Relative position from the end-effector to the target
        """
        dof_pos_scaled = 2.0 * (self._robot.data.joint_pos - self._robot.data.default_joint_pos)

        ee_pos = self._ee_frame.data.target_pos_w[..., 0, :]
        self.scene.ee_markers.visualize(ee_pos)

        root_pos = self._robot.data.root_state_w[:, :3]
        target = self.object.data.root_pos_w[:, :3]
        to_target = target - ee_pos

        obs = torch.cat(
            (dof_pos_scaled, self._robot.data.joint_vel, to_target),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        """Calculate the step reward based on distance to the target and action penalty."""
        target = self.object.data.root_pos_w[:, :3]
        ee_pos = self._ee_frame.data.target_pos_w[..., 0, :]
        d = torch.norm(ee_pos - target, p=2, dim=-1)
        # Distance-based reward (higher for being close)
        reach_object_reward = self.cfg.dist_reward_scale / (1.0 + d**2)


        lift_object_reward = self.cfg.lift_reward_scale * torch.where(self.object.data.root_pos_w[:, 2] > 0.04, 1.0, 0.0)

        target_height_dist = self.object.data.root_pos_w[:, 2] - 0.3
        target_height_reward = (self.object.data.root_pos_w[:, 2] > 0.04) * (1 - torch.tanh(target_height_dist / 0.3))
        target_height_reward *= self.cfg.height_reward_scale

        # Action penalty for smoothness
        action_penalty = torch.sum(self.actions**2, dim=-1) * self.cfg.action_penalty_scale

        self.extras["log"] = {
            "distance": (d).mean(),
            "object_height": (target_height_dist).mean(),
            "lift_object_reward": (lift_object_reward).mean(),
            "reach_object_reward": (reach_object_reward).mean(),
            "target_height_reward": (target_height_reward).mean(),
            "action_penalty": (action_penalty).mean(),
        }

        return reach_object_reward + lift_object_reward + target_height_reward - action_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine whether an episode should terminate or truncate."""
        object_height = self.object.data.root_pos_w[:, 2]
        # Terminate if close enough to the goal
        terminated = object_height > 0.3

        # Truncate if max episode length is reached
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset environments with the provided indices."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Randomize joint positions
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            self.cfg.initial_joint_pos_range[0],
            self.cfg.initial_joint_pos_range[1],
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        # Set joint positions and velocities
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # Randomize target position
        self.target_pos[env_ids, 0] = sample_uniform(0.35, 0.65, (len(env_ids),), self.device)
        self.target_pos[env_ids, 1] = sample_uniform(-0.2, 0.2, (len(env_ids),), self.device)
        self.target_pos[env_ids, 2] = sample_uniform(0.15, 0.5, (len(env_ids),), self.device)