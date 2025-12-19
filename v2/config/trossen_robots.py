import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.math import quat_from_euler_xyz
import torch
import math
import os

TROSSEN_ROBOT_CONFIGS = {
    "wx250s": {
        "articulation_cfg": ArticulationCfg(
            spawn = sim_utils.UsdFileCfg(usd_path = f"{os.path.dirname(__file__)}/../usd/robots/wx250s.usd"),
            prim_path = "",
            init_state = ArticulationCfg.InitialStateCfg(
                joint_pos = {"left_finger": 0.015, "right_finger": -0.015}
            ),
            actuators = {
                "waist_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["waist"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "shoulder_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["shoulder"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "elbow_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["elbow"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "forearm_roll_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["forearm_roll"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "wrist_angle_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["wrist_angle"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "wrist_rotate_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["wrist_rotate"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "gripper_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["gripper"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "left_finger_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["left_finger"],
                    stiffness = 600.0,
                    damping = 5.0
                ),
                "right_finger_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["right_finger"],
                    stiffness = 600.0,
                    damping = 5.0
                ),
            }
        ),
        "starting_pos": [0.0, -0.96, 1.16, 0.0, -0.3, 0.0] # This starting position is obtained from the trossen source code. The arms start recording data from this position.
    },
    "vx300s": {
        "articulation_cfg": ArticulationCfg(
            spawn = sim_utils.UsdFileCfg(usd_path = f"{os.path.dirname(__file__)}/../usd/robots/vx300s.usd"),
            prim_path = "",
            init_state = ArticulationCfg.InitialStateCfg(
                joint_pos = {"left_finger": 0.021, "right_finger": -0.021}
            ),
            actuators = {
                "waist_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["waist"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "shoulder_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["shoulder"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "elbow_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["elbow"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "forearm_roll_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["forearm_roll"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "wrist_angle_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["wrist_angle"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "wrist_rotate_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["wrist_rotate"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "gripper_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["gripper"],
                    stiffness = 600.0,
                    damping = 10.0
                ),
                "left_finger_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["left_finger"],
                    stiffness = 600.0,
                    damping = 5.0
                ),
                "right_finger_actuator": ImplicitActuatorCfg(
                    joint_names_expr = ["right_finger"],
                    stiffness = 600.0,
                    damping = 5.0
                ),
            }
        ),
        "starting_pos": [0.0, -0.96, 1.16, 0.0, -0.3, 0.0] # This starting position is obtained from the trossen source code. The arms start recording data from this position.
    }
}


def get_robot_articulation_cfg(robot_name, prim_path, pos = (0.0, 0.0, 0.0), rot = (0.0, 0.0, 0.0)):
    rot = quat_from_euler_xyz(torch.tensor(math.radians(rot[0])), torch.tensor(math.radians(rot[1])), torch.tensor(math.radians(rot[2])))
    articulation_cfg = TROSSEN_ROBOT_CONFIGS[robot_name]["articulation_cfg"].copy()
    articulation_cfg.prim_path = prim_path
    articulation_cfg.init_state.pos = pos
    articulation_cfg.init_state.rot = rot
    return articulation_cfg

def get_robot_starting_pos(robot_name):
    return TROSSEN_ROBOT_CONFIGS[robot_name]["starting_pos"]