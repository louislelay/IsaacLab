# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject, AssetBase
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def reset_jenga(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]] = {},
):


    if env_ids is None:
        return

    for cur_env in env_ids.tolist():
        # create a random position variance
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        pose = [random.uniform(range[0], range[1]) for range in range_list]

        # Split the pose list into position (x, y, z) and Euler angles (roll, pitch, yaw)
        position = pose[:3]  # x, y, z
        euler_angles = pose[3:]  # roll, pitch, yaw

        # Convert Euler angles (roll, pitch, yaw) to quaternion using the torch-based function
        roll, pitch, yaw = torch.tensor([euler_angles[0]]), torch.tensor([euler_angles[1]]), torch.tensor([euler_angles[2]])
        quaternion = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # Convert quaternion to a list
        quaternion_list = quaternion.squeeze().tolist()

        # Return the complete pose as a flat list: [x, y, z, x, y, z, w]
        pose_list = position + quaternion_list

        rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]
        
        object_state = rigid_object_collection.data.default_object_state.clone()
        object_state[..., :3] += env.scene.env_origins.unsqueeze(1)
        object_state = object_state[cur_env:cur_env+1]   # modify to get the current env

        for i in range(object_state.size(1)):
            object_state[0, i, :7] += torch.tensor(pose_list).to(object_state.device)

        rigid_object_collection.write_object_link_pose_to_sim(object_state[..., :7], env_ids=torch.tensor([cur_env], device=env.device))
        rigid_object_collection.write_object_com_velocity_to_sim(object_state[..., 7:], env_ids=torch.tensor([cur_env], device=env.device))