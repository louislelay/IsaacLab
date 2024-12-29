# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Disney Research robots.

The following configuration parameters are available:

* :obj:`BD1_CFG`: The BD-1 robot with DC actuator model
* :obj:`BD1_CFG`: The BD-X robot with implicit Actuator model

Reference:

* https://github.com/MoscowskyAnton/BD1/tree/main/bd1_description
* https://github.com/rimim/AWD/tree/main/awd/data/assets/go_bdx

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

BD1_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=["foot_.*", "knee_.*", "hip_.*", "neck_j", "head_j"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)

##
# Configuration
##

BD1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/louis/Documents/simulation/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/bd1/bd1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={".*": BD1_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Disney BD-1 robot with DC actuator model."""

BDX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/louis/Documents/simulation/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/bdx/bdx.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", ".*_ankle"],
            stiffness={ 
                ".*_hip_yaw": 100.0,
                ".*_hip_roll": 80.0,
                ".*_hip_pitch": 120.0,
                ".*_knee": 200.0,
                ".*_ankle": 200.0,
            },
            damping={
                ".*_hip_yaw": 3.0,
                ".*_hip_roll": 3.0,
                ".*_hip_pitch": 6.0,
                ".*_knee": 6.0,
                ".*_ankle": 6.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["neck_pitch", "head_pitch", "head_yaw", "head_roll", ".*_antenna"],
            stiffness={
                ".*": 20.0,
            },
            damping={
                ".*": 1.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Disney BD-X robot with implicit actuator model."""