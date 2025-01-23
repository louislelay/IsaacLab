# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# def objects_position_in_robot_root_frame(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
# ) -> torch.Tensor:
#     """The position of the object in the robot's root frame."""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     # object: RigidObject = env.scene[object_cfg.name]
#     rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

#     num_piece = 2 # select one of the piece of thge jenbga
#     object_pos_w = rigid_object_collection.data.object_state_w[:, num_piece, :3]

#     # object_pos_w = rigid_object_collection.data.object_link_state_w[:, :3]
#     object_pos_b, _ = subtract_frame_transforms(
#         robot.data.root_link_state_w[:, :3], robot.data.root_link_state_w[:, 3:7], object_pos_w
#     )

#     print("object_pos_b")
#     print(object_pos_b)
#     print(object_pos_b.shape)
#     return object_pos_b





def objects_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
) -> torch.Tensor:
    """The positions of all objects in the robot's root frame."""
    # Retrieve the robot and the collection of rigid objects
    robot: RigidObject = env.scene[robot_cfg.name]
    rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    # Extract positions and orientations for the robot
    robot_pos_w = robot.data.root_link_state_w[:, :3]  # Shape: (N_envs, 3)
    robot_orient_w = robot.data.root_link_state_w[:, 3:7]  # Shape: (N_envs, 4)

    # Extract positions of all objects in the world frame
    object_pos_w = rigid_object_collection.data.object_state_w[:, :, :3]  # Shape: (N_envs, N_objects, 3)

    # Compute positions of all objects in the robot's root frame
    # Iterate over the second dimension (objects) and compute transformations
    N_envs, N_objects, _ = object_pos_w.shape
    object_pos_b = torch.empty_like(object_pos_w)  # Placeholder for results

    for i in range(N_objects):
        # Extract the i-th object's position in the world frame
        obj_pos_w = object_pos_w[:, i, :]  # Shape: (N_envs, 3)

        print("obj_pos_w")
        print(obj_pos_w)

        print("robot_pos_w")
        print(robot_pos_w)

        print("robot_orient_w")
        print(robot_orient_w)

        # Transform the i-th object's position into the robot's root frame
        obj_pos_b, _ = subtract_frame_transforms(
            robot_pos_w,
            robot_orient_w,
            obj_pos_w
        )
        print("obj_pos_b")
        print(obj_pos_b)
        object_pos_b[:, i, :] = obj_pos_b

    # Resulting shape: (N_envs, N_objects, 3)
    print("object_pos_b")
    print(object_pos_b)
    print(object_pos_b.shape)

    # To make it be concatenate
    transformed_tensor = object_pos_b.reshape(2, -1)
    print("transformed_tensor")
    print(transformed_tensor)
    print(transformed_tensor.shape)

    return transformed_tensor
