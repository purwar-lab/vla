import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.math import quat_from_euler_xyz
import torch
import math
import os

ITEM_PROPS = {
    "blue_prime_bottle": {
        "mass": 0.5,
        "boundary_threshold": 0.05
    },
    "chips_packet": {
        "mass": 0.05,
        "boundary_threshold": 0.08
    },
    "croissant": {
        "mass": 0.1,
        "boundary_threshold": 0.05
    },
    "fanta_can": {
        "mass": 0.3,
        "boundary_threshold": 0.04
    },
    "green_prime_bottle": {
        "mass": 0.5,
        "boundary_threshold": 0.05
    },
    "soda_can": {
        "mass": 0.3,
        "boundary_threshold": 0.04
    },
    "toy_car": {
        "mass": 0.3,
        "boundary_threshold": 0.1
    },
    "red_cube": {
        "mass": 0.1,
        "boundary_threshold": 0.032
    },
    "green_cube": {
        "mass": 0.1,
        "boundary_threshold": 0.032
    },
    "blue_cube": {
        "mass": 0.1,
        "boundary_threshold": 0.032
    },
    "red_cuboid": {
        "mass": 0.1,
        "boundary_threshold": 0.02
    },
    "green_cuboid": {
        "mass": 0.1,
        "boundary_threshold": 0.02
    },
    "blue_cuboid": {
        "mass": 0.1,
        "boundary_threshold": 0.02
    },
    "cardboard_box": {
        "mass": 0.5,
        "boundary_threshold": 0.91
    }
}

def get_item_config(item_name, prim_path = None, mass = None, pos = (0.0, 0.0, 0.0), rot = (0.0, 0.0, 0.0)):
    if prim_path == None:
        prim_path = "{ENV_REGEX_NS}/" + item_name
    if mass == None:
        mass = ITEM_PROPS[item_name]["mass"]
    rot = quat_from_euler_xyz(torch.tensor(math.radians(rot[0])), torch.tensor(math.radians(rot[1])), torch.tensor(math.radians(rot[2])))
    object_cfg = RigidObjectCfg(
        prim_path = prim_path,
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"{os.path.dirname(__file__)}/../usd/items/{item_name}/asset.usd",
            rigid_props = sim_utils.RigidBodyPropertiesCfg(),
            mass_props = sim_utils.MassPropertiesCfg(mass = mass),
            collision_props = sim_utils.CollisionPropertiesCfg()
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos = pos, rot = rot)
    )
    return object_cfg