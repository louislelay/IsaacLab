# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg, FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.jenga import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.jenga.jenga_env_cfg import JengaEnvCfg

##
# Pre-defined configs
##

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

@configclass
class FrankaJengaEnvCfg(JengaEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.commands.object_pose.body_name = "panda_hand"

        # Block dimensions
        block_length = 0.15  # Length of a single block
        block_width = 0.05   # Width of a single block
        block_height = 0.03  # Height of a single block

        # Starting position for the first block
        start_pos = [0.5, 0.0, 0.0120]  # Adjust z to align blocks correctly

        # Define colors for the blocks
        colors = [
            (0.76, 0.60, 0.42),  # First color
            (0.3, 0.60, 0.42),   # Second color
            (0.5, 0.5, 0.5),     # Third color
        ]

        # Initialize rigid_objects dictionary
        rigid_objects = {}

        # Number of layers and blocks per layer
        num_layers = 1 #6
        blocks_per_layer = 1 #3

        # Create the Jenga tower
        for layer in range(num_layers):
            is_horizontal = layer % 2 == 0  # Alternate between horizontal and vertical
            for i in range(blocks_per_layer):
                object_name = f"object_{layer}_{i}"  # Unique name for each block

                # Determine position and orientation
                if is_horizontal:
                    x_offset = start_pos[0] 
                    y_offset = start_pos[1] + i * block_width

                    x_size = block_length
                    y_size = block_width
                else:
                    x_offset = start_pos[0] + (i-1) * block_width
                    y_offset = start_pos[1] + block_width

                    x_size = block_width
                    y_size = block_length

                z_offset = start_pos[2] + layer * block_height

                # Determine color
                color = colors[i % len(colors)]

                # Add to rigid_objects dictionary
                rigid_objects[object_name] = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/{object_name}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[x_offset, y_offset, z_offset], rot=[1, 0, 0, 0]),
                    spawn=sim_utils.MeshCuboidCfg(
                        size=(x_size, y_size, block_height),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(density=700.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
                    ),
                )

        # Assign to the object collection
        self.scene.object_collection = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
