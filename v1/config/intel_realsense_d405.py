import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_from_euler_xyz
import torch
import math

CAMERA_PROPERTIES = {
    "640x480": {
        "Model": "Intel RealSense D405",
        "FPS": 90,
        "Width": 640,
        "Height": 480,
        "PPX": 327.656982421875,
        "PPY": 241.532531738281,
        "Fx": 434.815490722656,
        "Fy": 434.216217041016
    },
    "848x480": {
        "Model": "Intel RealSense D405",
        "FPS": 90,
        "Width": 848,
        "Height": 480,
        "PPX": 423.902435302734,
        "PPY": 242.607940673828,
        "Fx": 433.806182861328,
        "Fy": 433.274871826172
    }
}


def get_camera_cfg(prim_path, pos = (0.0, 0.0, 0.0), rot = (0.0, 0.0, 0.0), resolution = "848x480", data_types = ["rgb"]):
    camera_properties = CAMERA_PROPERTIES[resolution]
    camera_intrinsic_matrix = [camera_properties["Fx"], 0, camera_properties["PPX"], 0, camera_properties["Fy"], camera_properties["PPY"], 0, 0, 1]
    rot = quat_from_euler_xyz(torch.tensor(math.radians(rot[0])), torch.tensor(math.radians(rot[1])), torch.tensor(math.radians(rot[2])))
    camera_cfg = CameraCfg(
        prim_path = prim_path,
        update_period = 1/camera_properties["FPS"],
        height = camera_properties["Height"],
        width = camera_properties["Width"],
        data_types = data_types,
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(camera_intrinsic_matrix, height = camera_properties["Height"], width = camera_properties["Width"]),
        offset = CameraCfg.OffsetCfg(pos = pos, rot = rot , convention = "world")
    )
    return camera_cfg

