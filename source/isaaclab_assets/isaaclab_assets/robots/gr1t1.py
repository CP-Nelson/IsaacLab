# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`Gr1t1_LeftArm_CFG`: Gr1t1 left arm configuration

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

Gr1t1_LeftArm_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/GR1T1/GR1T1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "l_shoulder_pitch": 0.0,
            "l_shoulder_roll": 0.2,
            "l_shoulder_yaw": 0.0,
            "l_elbow_pitch": -0.3,
            "l_wrist_yaw": 0.0,
            "l_wrist_roll": 0.0,
            "l_wrist_pitch": 0.0,
        },
    ),
    actuators={
        "shoulder_pitch": ImplicitActuatorCfg(
            joint_names_expr=["l_shoulder_pitch"],
            effort_limit=500,
            velocity_limit=2.5,
            stiffness=92.85,
            damping=2.575,
        ),
        "shoulder_roll": ImplicitActuatorCfg(
            joint_names_expr=["l_shoulder_roll"],
            effort_limit=500,
            velocity_limit=2.5,
            stiffness=92.85,
            damping=2.575,
        ),
        "shoulder_yaw": ImplicitActuatorCfg(
            joint_names_expr=["l_shoulder_yaw"],
            effort_limit=500,
            velocity_limit=2.5,
            stiffness=112.06,
            damping=3.1,
        ),
        "elbow_pitch": ImplicitActuatorCfg(
            joint_names_expr=["l_elbow_pitch"],
            effort_limit=500,
            velocity_limit=2.5,
            stiffness=112.06,
            damping=3.1,
        ),
        "writst": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit=500,
            velocity_limit=2.5,
            stiffness=10.0,
            damping=1.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""