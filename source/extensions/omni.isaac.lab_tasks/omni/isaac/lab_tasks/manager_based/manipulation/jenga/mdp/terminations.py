# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def jenga_tower_fell(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Check if the Jenga tower has fallen based on the positions of its pieces.

    The function terminates the environment when any Jenga piece is below its 
    initial height. This is determined by comparing the current heights of 
    all pieces to their default (initial) heights.

    Args:
        env (ManagerBasedRLEnv): The reinforcement learning environment containing the scene.
        asset_cfg (SceneEntityCfg): Configuration for the Jenga pieces (asset) in the scene.

    Returns:
        torch.Tensor: A boolean tensor of shape (N,), where N is the number of environments.
                      True indicates the Jenga tower has fallen in the respective environment.
    """
    # Retrieve the Jenga pieces collection from the environment
    rigid_object_collection: RigidObjectCollection = env.scene[asset_cfg.name]
    
    # Compare the current height (z-axis) of each Jenga piece with its initial height
    # - `object_state_w[:, :, 2]`: Current height of all pieces across environments.
    # - `default_object_state[:, :, 2]`: Initial height of all pieces across environments.
    # - `result` shape: (N, M), where N is the number of environments and M is the number of pieces.
    result = rigid_object_collection.data.object_state_w[:, :, 2] < rigid_object_collection.data.default_object_state[:, :, 2]

    # Check if any piece in each environment is below its initial height
    # - `result.any(dim=1)`: Reduces the result along the pieces dimension (M) to (N,).
    result = result.any(dim=1)

    return result
