import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_from_euler_xyz
import torch
import math

def get_camera_cfg(prim_path, pos = (0.0, 0.0, 0.0), rot = (0.0, 0.0, 0.0), resolution = "640x480", focal_length = 24, data_types = ["rgb"]):
    # focal_length is in cm
    fps = 60
    resolution = resolution.split("x")
    width = int(resolution[0])
    height = int(resolution[1])
    rot = quat_from_euler_xyz(torch.tensor(math.radians(rot[0])), torch.tensor(math.radians(rot[1])), torch.tensor(math.radians(rot[2])))
    camera_cfg = CameraCfg(
        prim_path = prim_path,
        update_period = 1/fps,
        height = height,
        width = width,
        data_types = data_types,
        spawn = sim_utils.PinholeCameraCfg(focal_length=focal_length),
        offset = CameraCfg.OffsetCfg(pos = pos, rot = rot , convention = "world")
    )
    return camera_cfg

