import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
import os

def get_grid_cfg(prim_path):
    grid_cfg = AssetBaseCfg(
        prim_path = prim_path,
        spawn = sim_utils.UsdFileCfg(usd_path = f"{os.path.dirname(__file__)}/../usd/scenes/floor_grid/scene.usd")
    )
    return grid_cfg