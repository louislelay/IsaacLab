# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    target_object_id : int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]
    return torch.where(rigid_object_collection.data.object_link_state_w[:, target_object_id, 2] > minimal_height, 1.0, 0.0)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    target_object_id : int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    object_pos_w = rigid_object_collection.data.object_link_state_w[:, target_object_id, :3]  # Shape: (N_envs, N_objects, 3)
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    target_object_id : int = 2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_link_state_w[:, :3], robot.data.root_link_state_w[:, 3:7], des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - rigid_object_collection.data.object_link_state_w[:, target_object_id, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (rigid_object_collection.data.object_link_state_w[:, target_object_id, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def tower_stability_reward_acceleration(
    env: ManagerBasedRLEnv,
    target_object_id: int,
    acceleration_threshold: float,
    instability_penalty: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
) -> torch.Tensor:
    """
    Reward function to evaluate tower stability based on object acceleration.
    Penalizes instability for objects other than the target object.
    """
    # Retrieve the collection of rigid objects
    rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    # Linear acceleration of all objects in the world frame: (num_envs, num_objects, 3)
    object_acc_w = rigid_object_collection.data.object_acc_w[:, :, :3]

    # Identify objects with acceleration above the threshold (excluding the target block)
    unstable = object_acc_w > acceleration_threshold

    unstable[:, target_object_id, :] = False  # Exclude the target block

    # Calculate the penalty for unstable objects (sum across objects and dimensions)
    instability_count = torch.sum(unstable.float(), dim=(1, 2))

    # Convert instability count to a reward penalty
    reward = -instability_count * instability_penalty

    return reward