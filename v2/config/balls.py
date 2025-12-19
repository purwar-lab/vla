import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

COMMON_BALL_PROPERTIES = {
    "radius": 0.03111,
    "colors" : {
        "red": [0.654, 0.137, 0.123],
        "green": [0.185, 0.629, 0.12],
        "blue": [0.006, 0.138, 0.652]
    }
}

def get_ball_cfg(color, prim_path, pos = None, radius = COMMON_BALL_PROPERTIES["radius"]):
    if pos == None:
        init_pos = (0.6, 0, radius)
    else:
        init_pos = pos

    if type(color) is str:
        color = tuple(COMMON_BALL_PROPERTIES["colors"].get(color, (1.0, 1.0, 1.0)))
    
    ball_cfg = RigidObjectCfg(
        prim_path = prim_path,
        spawn = sim_utils.SphereCfg(
                radius = radius,
                visual_material = sim_utils.PreviewSurfaceCfg(roughness = 1, diffuse_color = color),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(),
                mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
                collision_props = sim_utils.CollisionPropertiesCfg()
            ),
        init_state = RigidObjectCfg.InitialStateCfg(pos = init_pos),
    )
    return ball_cfg