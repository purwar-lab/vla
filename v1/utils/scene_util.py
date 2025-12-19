import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
import torch
import math
from copy import deepcopy
import random
import time
import h5py
import cv2
import numpy as np

import os
script_directory = f"{os.path.dirname(__file__)}"
import sys
from pathlib import Path
sys.path.append(str(Path(script_directory).parent))
from config import trossen_robots
from config import cameras
from config import custom_items
from utils import custom_util

MAX_STEPS = 250
MAX_SUB_STEPS = 50

ROBOT_NAME = "vx300s"

GRIPPER_BUFFER = 0.06
GRIPPER_ACTIONS = {
    "close": 0,
    "open": 1
}

KITCHEN_SCENE_TARGET_POSITIONS = [
    [0.45, 0.25, 0.876],
    [0.45, 0, 0.876],
    [0.45, -0.25, 0.876]
]

IK_LOSS_THRESHOLD = 0.001
CAPTURE_DATA_STEP_INTERVAL = 2

PERSPECTIVE_CAMERA_POSITION = [-2.0, 0, 2.5]
PERSPECTIVE_CAMERA_LOOK_AT = [2.0, 0.0, 1.0]

ITEM_NAMES = ["red_cuboid", "green_cuboid", "blue_cuboid", "cardboard_box"]

PLATFORM = "cardboard_box"

@configclass
class CustomSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(prim_path = "/World/defaultGroundPlane", spawn = sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path = "/World/Light", spawn = sim_utils.DomeLightCfg(intensity = 5000.0, color = (1.0, 1.0, 1.0))
    )

    kitchen = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/Kitchen",
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"{os.path.dirname(__file__)}/../usd/scenes/kitchen_scene/scene.usd",
            rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True), # kinematic_enabled = True
            mass_props = sim_utils.MassPropertiesCfg(mass = 1000),
            collision_props = sim_utils.CollisionPropertiesCfg(),
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos = (1.1, 0.3, 0), rot = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), torch.tensor(math.radians(-45))))
    )

    # articulation
    robot = trossen_robots.get_robot_articulation_cfg(robot_name = ROBOT_NAME, prim_path = "{ENV_REGEX_NS}/Robot", pos = (0, 0, 0.88))
    
    # camera_wrist_right
    camera_wrist_right = cameras.get_camera_cfg(prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_NAME + "_gripper_bar_link/camera_wrist_right", resolution = "300x300", focal_length = 10, pos = (-0.035, 0.01, 0.1), rot = (0, 20, 0))

    # camera_high
    camera_high = cameras.get_camera_cfg(prim_path = "{ENV_REGEX_NS}/Robot/camera_high", resolution = "300x300", pos = (1.4, 0, 0.65), rot = (0, 30, -180))

    red_cuboid = custom_items.get_item_config(item_name = "red_cuboid", mass = 0.05, pos = (20, 4, 0.06))
    green_cuboid = custom_items.get_item_config(item_name = "green_cuboid", mass = 0.05, pos = (20, 10, 0.06))
    blue_cuboid = custom_items.get_item_config(item_name = "blue_cuboid", mass = 0.05, pos = (20, 12, 0.06))

    cardboard_box = RigidObjectCfg(
        prim_path = "{ENV_REGEX_NS}/" + "cardboard_box",
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"{os.path.dirname(__file__)}/../usd/items/cardboard_box/asset.usd",
            rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled = True),
            collision_props = sim_utils.CollisionPropertiesCfg()
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos = (20, 14, 0.06))
    )


class CustomScene:
    def __init__(self, args_cli):
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
        self.sim = SimulationContext(sim_cfg)
        # Set main camera
        self.sim.set_camera_view(PERSPECTIVE_CAMERA_POSITION, PERSPECTIVE_CAMERA_LOOK_AT)
        # Design scene
        scene_cfg = CustomSceneCfg(num_envs = 1, env_spacing = 5.0) # Hardcoding it to use only 1 env. Multi envs have some weird issue that I don't need to resolve right now
        self.scene = InteractiveScene(scene_cfg)
        # Play the simulator
        self.sim.reset()
        # Now we are ready!

        self.sim_dt = self.sim.get_physics_dt()

        self.robot = self.scene["robot"]

        self.gripper_pos_limits = self.robot.data.soft_joint_pos_limits[0, 7].clone().cpu().numpy().tolist()

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)

        self.ik_target = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.robot.device)

        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"], body_names=[f"{ROBOT_NAME}_fingers_link"])
        self.robot_entity_cfg.resolve(self.scene)

        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        self.target_joint_pos = None
        print("[INFO]: Setup complete...")


    def initialize_captured_data_dict(self):
        self.captured_data = {
            "/observations/x_y_z_roll_pitch_yaw_gripper": [],
            "/observations/qpos": [],
            "/observations/images/camera_high": [],
            "/observations/images/camera_wrist_right": []
        }

    
    def place_items_randomly(self, episode_index=0, capture_data_flag=False):
        self.initialize_captured_data_dict()
        
        all_item_names = deepcopy(ITEM_NAMES[:3])
        target_item_name = all_item_names[episode_index % len(all_item_names)]
        random.shuffle(all_item_names)

        task = f"Pick up the {target_item_name.replace('_', ' ')} and place it on the {PLATFORM.replace('_', ' ')}"
        
        initial_item_positions = {}
        initial_item_orientations = {}

        for i in range(len(all_item_names)):
            item_name = all_item_names[i]
            [x, y, z] = KITCHEN_SCENE_TARGET_POSITIONS[i]
            # x = random.uniform(x-0.075, x+0.075)
            y = random.uniform(y-0.05, y+0.05)
            x_y_z_roll_pitch_yaw = [x, y, z, 0, 0, 0]
            self.place_item_at_x_y_z_roll_pitch_yaw(item_name, x_y_z_roll_pitch_yaw)

            initial_item_positions[item_name] = deepcopy(x_y_z_roll_pitch_yaw[:3])
            initial_item_orientations[item_name] = deepcopy(x_y_z_roll_pitch_yaw[3:])
        
        [x, y, z] = deepcopy(initial_item_positions[target_item_name])
        x += 0.24
        y += random.uniform(-0.1, 0.1)
        x_y_z_roll_pitch_yaw = [x, y, z, 0, 0, 0]
        self.place_item_at_x_y_z_roll_pitch_yaw(PLATFORM, x_y_z_roll_pitch_yaw)
        initial_item_positions[PLATFORM] = deepcopy(x_y_z_roll_pitch_yaw[:3])
        initial_item_orientations[PLATFORM] = deepcopy(x_y_z_roll_pitch_yaw[3:])

        joint_pos = self.robot.data.default_joint_pos.clone() # Just to get the format
        joint_pos[0, :6] = torch.tensor(trossen_robots.get_robot_starting_pos(ROBOT_NAME))
        # joint_pos[0, 4] = 0.2 # Pointing the gripper down to see the pick_up_items
        
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.robot.reset()

        # Go to random starting point
        x = random.uniform(0.2, 0.3)
        y = random.uniform(-0.2, 0.2)
        z = random.uniform(1.0, 1.2)
        
        look_x = 0.6
        look_y = random.uniform(-0.1, 0.1)
        look_z = random.uniform(0.89, 1.1)

        distance = custom_util.get_distance_between_two_points([x, y, z], [look_x, look_y, look_z])
        
        pitch = math.asin(abs(z - look_z)/distance)
        if look_z > z:
            pitch = -pitch
        
        yaw = math.asin(abs(y - look_y)/distance)
        if look_y < y:
            yaw = -yaw
        
        roll = random.uniform(-0.35, 0.35)

        initial_x_y_z_roll_pitch_yaw_gripper = [x, y, z, roll, pitch, yaw, GRIPPER_ACTIONS["open"]]

        self.move_ee_to_pos(initial_x_y_z_roll_pitch_yaw_gripper, max_steps=500, world=True)

        self.robot.reset()

        for _ in range(2): # Multiple steps as workaround to update the camera and other sensors and prevent stale data
            self.update_scene()

        self.scene_info = {}
        self.scene_info["task"] = task
        self.scene_info["target_item_name"] = target_item_name
        self.scene_info["initial_item_positions"] = initial_item_positions
        self.scene_info["initial_item_orientations"] = initial_item_orientations
        self.scene_info["initial_x_y_z_roll_pitch_yaw_gripper"] = initial_x_y_z_roll_pitch_yaw_gripper

        [grab_x, grab_y, grab_z] = self.get_front_center_of_item(target_item_name)

        #randomize target from center
        center_delta = 0.007
        grab_y += random.uniform(-center_delta, center_delta)
        grab_z += 0.04 + random.uniform(-center_delta, center_delta)

        radian_delta = math.radians(3)
        grab_roll = random.uniform(-radian_delta, radian_delta)
        grab_pitch = random.uniform(-radian_delta, radian_delta)
        grab_yaw = 0 #random.uniform(-radian_delta, radian_delta)

        self.scene_info["grab_x_y_z_roll_pitch_yaw_gripper"] = [grab_x, grab_y, grab_z, grab_roll, grab_pitch, grab_yaw, GRIPPER_ACTIONS["close"]]
        

        [place_x, place_y, place_z] = deepcopy(initial_item_positions[PLATFORM])
        place_x += -0.04 + random.uniform(-0.02, 0.02)
        place_y += random.uniform(-0.02, 0.02)
        place_z += 0.17 + random.uniform(0.00, 0.01)
        place_roll = random.uniform(-radian_delta, radian_delta)
        place_pitch = random.uniform(-radian_delta, radian_delta)
        place_yaw = 0 #random.uniform(-radian_delta, radian_delta)

        self.scene_info["place_x_y_z_roll_pitch_yaw_gripper"] = [place_x, place_y, place_z, place_roll, place_pitch, place_yaw, GRIPPER_ACTIONS["open"]]
        
        if capture_data_flag:
            self.capture_data()
        
        return self.scene_info


    def place_items_from_hdf5_file(self, hdf5_file_path, capture_data_flag=False):
        self.initialize_captured_data_dict()
        self.scene_info = {}

        with h5py.File(hdf5_file_path, "r") as hdf5_fobj:
            target_item_name = hdf5_fobj.attrs["target_item_name"]
            task = hdf5_fobj.attrs["task"]
            initial_item_positions = {}
            initial_item_orientations = {}
            for item_name in ITEM_NAMES:
                x_y_z = hdf5_fobj.attrs[f"{item_name}_position"].tolist()
                roll_pitch_yaw = hdf5_fobj.attrs[f"{item_name}_orientation"].tolist()
                self.place_item_at_x_y_z_roll_pitch_yaw(item_name, x_y_z + roll_pitch_yaw)
                initial_item_positions[item_name] = x_y_z
                initial_item_orientations[item_name] = roll_pitch_yaw

            initial_qpos = hdf5_fobj["observations"]["qpos"][0]
            self.target_joint_pos = self.robot.data.default_joint_pos.clone() # Just to get the format
            self.target_joint_pos[0, :6] = torch.tensor(initial_qpos[:6])
            gripper_action = self.unnormalize_gripper_action(initial_qpos[6])
            self.target_joint_pos[0, 7] = gripper_action
            self.target_joint_pos[0, 8] = -gripper_action            
            joint_vel = self.robot.data.default_joint_vel.clone()
            self.robot.write_joint_state_to_sim(self.target_joint_pos, joint_vel)

            self.robot.reset()
            
            for _ in range(2): # Multiple steps as workaround to update the camera and other sensors and prevent stale data
                self.update_scene()

            self.scene_info["task"] = task
            self.scene_info["target_item_name"] = target_item_name
            self.scene_info["initial_item_positions"] = initial_item_positions
            self.scene_info["initial_item_orientations"] = initial_item_orientations
        
        if capture_data_flag:
            self.capture_data()
        
        return self.scene_info


    def get_front_center_of_item(self, item_name):
        [item_x, item_y, item_z] = self.scene_info["initial_item_positions"][item_name]
        x_y_z = [0, 0, 0]
        x_y_z[0] = item_x - (custom_items.ITEM_PROPS[item_name]["boundary_threshold"])
        x_y_z[1] = item_y
        x_y_z[2] = item_z + 0.018
        return x_y_z
    

    def initial_pause_for_lighting(self):
        wait_time = 5 # Wait for x seconds before starting to record. This is to help stabilize the lighting
        print(f"Waiting for {wait_time} seconds before starting...")
        wait_start_time = time.time()
        while (time.time() - wait_start_time) < wait_time:
            self.update_scene()
    

    def update_scene(self):
        if self.target_joint_pos != None:
            self.robot.set_joint_position_target(self.target_joint_pos)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim_dt)


    def convert_from_world_frame_to_robot_frame(self, world_pos, world_quat):
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        rel_pos, rel_quat = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], world_pos, world_quat
        )
        return rel_pos, rel_quat


    def get_ee_pos_and_quat_with_respect_to_robot(self):
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        return self.convert_from_world_frame_to_robot_frame(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
    

    def normalize_gripper_action(self, gripper_action):
        normalized_gripper_action = (gripper_action - self.gripper_pos_limits[0]) / (self.gripper_pos_limits[1] - self.gripper_pos_limits[0])
        return normalized_gripper_action


    def unnormalize_gripper_action(self, normalized_gripper_action):
        gripper_action = self.gripper_pos_limits[0] + (normalized_gripper_action * (self.gripper_pos_limits[1] - self.gripper_pos_limits[0]))
        return gripper_action


    def calculate_joint_pos_from_ik(self):
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        joint_pos = self.robot.data.joint_pos.clone()
        target_joint_pos = joint_pos.clone()
        ee_pos_b, ee_quat_b = self.get_ee_pos_and_quat_with_respect_to_robot()
        temp_target_joint_pos = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos[:, self.robot_entity_cfg.joint_ids])
        target_joint_pos[:, self.robot_entity_cfg.joint_ids] = temp_target_joint_pos
        return target_joint_pos


    def move_ee_to_pos(self, x_y_z_roll_pitch_yaw_gripper, max_steps=MAX_SUB_STEPS, world=True, capture_data_flag=False):
        x_y_z = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)
        x_y_z[:, 0] = x_y_z_roll_pitch_yaw_gripper[0]
        x_y_z[:, 1] = x_y_z_roll_pitch_yaw_gripper[1]
        x_y_z[:, 2] = x_y_z_roll_pitch_yaw_gripper[2]

        quat = torch.zeros(self.scene.num_envs, 4, device=self.sim.device)
        quat[:] = quat_from_euler_xyz(torch.tensor(x_y_z_roll_pitch_yaw_gripper[3]), torch.tensor(x_y_z_roll_pitch_yaw_gripper[4]), torch.tensor(x_y_z_roll_pitch_yaw_gripper[5]))
        if world:
            relative_x_y_z, relative_quat = self.convert_from_world_frame_to_robot_frame(x_y_z, quat)
        else:
            relative_x_y_z = x_y_z
            relative_quat = quat
        self.ik_target[0, :] = torch.tensor([*relative_x_y_z[0], *relative_quat[0]], device=self.sim.device)
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(self.ik_target)

        normalized_gripper_action = x_y_z_roll_pitch_yaw_gripper[6]
        gripper_action = self.unnormalize_gripper_action(normalized_gripper_action)
        
        previous_joint_pos = self.robot.data.joint_pos.clone()

        for step_index in range(1, max_steps+1):
            self.target_joint_pos = self.calculate_joint_pos_from_ik()
            self.target_joint_pos[:, 7] = gripper_action
            self.target_joint_pos[:, 8] = -gripper_action
            self.update_scene()

            current_joint_pos = self.robot.data.joint_pos.clone()
            ik_loss = (previous_joint_pos[0, :] - current_joint_pos[0, :]).abs().max()
            if capture_data_flag and (ik_loss < IK_LOSS_THRESHOLD or step_index % CAPTURE_DATA_STEP_INTERVAL == 0):
                self.capture_data()
            if ik_loss < IK_LOSS_THRESHOLD:
                break
            previous_joint_pos = current_joint_pos
        
        for _ in range(10): # Stabilize robot
            self.update_scene()
        
        if capture_data_flag:
            self.capture_data()

    
    def apply_qpos_to_robot(self, qpos, max_steps=MAX_SUB_STEPS, capture_data_flag=False):
        self.target_joint_pos = self.robot.data.joint_pos.clone()
        self.target_joint_pos[:, :6] = torch.tensor(qpos[:6])
        normalized_gripper_action = qpos[6]
        gripper_action = self.unnormalize_gripper_action(normalized_gripper_action)
        self.target_joint_pos[:, 7] = gripper_action
        self.target_joint_pos[:, 8] = -gripper_action

        previous_joint_pos = self.robot.data.joint_pos.clone()

        for step_index in range(1, max_steps+1):
            self.update_scene()
            
            current_joint_pos = self.robot.data.joint_pos.clone()
            
            # I know this isn't ik, but still using the same methodology to make sure it reaches the correct qpos
            ik_loss = (previous_joint_pos[0, :] - current_joint_pos[0, :]).abs().max()
            
            if capture_data_flag and (ik_loss < IK_LOSS_THRESHOLD or step_index % CAPTURE_DATA_STEP_INTERVAL == 0):
                self.capture_data()
            
            if ik_loss < IK_LOSS_THRESHOLD:
                break
            previous_joint_pos = current_joint_pos
        
        for _ in range(10): # Stabilize robot
            self.update_scene()
        
        if capture_data_flag:
            self.capture_data()
    

    def operate_gripper(self, normalized_gripper_action, capture_data_flag=False):
        if self.target_joint_pos == None:
            self.target_joint_pos = self.robot.data.joint_pos.clone()
        gripper_action = self.unnormalize_gripper_action(normalized_gripper_action)
        self.target_joint_pos[:, 7] = gripper_action
        self.target_joint_pos[:, 8] = -gripper_action

        for _ in range(10):
            self.update_scene()

        if capture_data_flag:
            self.capture_data()


    def get_current_x_y_z_roll_pitch_yaw_of_item(self, item_name):
        x_y_z_roll_pitch_yaw = [0, 0, 0, 0, 0, 0]
        item_state = self.scene[item_name].data.root_state_w.clone()
        x_y_z_roll_pitch_yaw[:3] = item_state[0, :3].cpu().numpy().tolist()
        item_roll_pitch_yaw = euler_xyz_from_quat(item_state[:, 3:7])
        x_y_z_roll_pitch_yaw[3:] = [custom_util.convert_reflex_angle_to_negative_angle(x[0].item()) for x in item_roll_pitch_yaw]
        return x_y_z_roll_pitch_yaw
    

    def place_item_at_x_y_z_roll_pitch_yaw(self, item_name, x_y_z_roll_pitch_yaw):
        item = self.scene[item_name]
        item_state = item.data.default_root_state.clone()
        item_state[:, 0] = x_y_z_roll_pitch_yaw[0]
        item_state[:, 1] = x_y_z_roll_pitch_yaw[1]
        item_state[:, 2] = x_y_z_roll_pitch_yaw[2]
        item_state[:, 3:7] = quat_from_euler_xyz(torch.tensor(x_y_z_roll_pitch_yaw[3]), torch.tensor(x_y_z_roll_pitch_yaw[4]), torch.tensor(x_y_z_roll_pitch_yaw[5]))
        item.write_root_state_to_sim(item_state)


    def get_current_observation(self):
        camera_wrist_right = self.scene["camera_wrist_right"]
        camera_high = self.scene["camera_high"]
        
        joint_pos = self.robot.data.joint_pos[0].clone().cpu().numpy()
        
        normalized_gripper_state = self.normalize_gripper_action(joint_pos[7])
        if normalized_gripper_state >= 0.9:
            normalized_gripper_state = 1.0
        else:
            normalized_gripper_state = 0.0
        
        qpos = [*joint_pos[:6], normalized_gripper_state]     

        ee_pos_b, ee_quat_b = self.get_ee_pos_and_quat_with_respect_to_robot()
        roll_pitch_yaw = list(euler_xyz_from_quat(ee_quat_b))
        for i in range(3):
            roll_pitch_yaw[i] = custom_util.convert_reflex_angle_to_negative_angle(roll_pitch_yaw[i][0].cpu().item())
        
        ## Sanity check
        # print(ee_pose_w[3:7])
        # print(quat_from_euler_xyz(torch.tensor(roll_pitch_yaw[0], device=sim.device), torch.tensor(roll_pitch_yaw[1], device=sim.device), torch.tensor(roll_pitch_yaw[2], device=sim.device)))
        
        x_y_z_roll_pitch_yaw_gripper = [*ee_pos_b[0].clone().cpu().numpy(), *roll_pitch_yaw, normalized_gripper_state]

        camera_wrist_right_data_tensor = camera_wrist_right.data.output["rgb"]
        camera_wrist_right_data_tensor = camera_wrist_right_data_tensor.view(camera_wrist_right_data_tensor.shape[1:])
        camera_wrist_right_data = camera_wrist_right_data_tensor.cpu().numpy()

        # camera_high
        camera_high_data_tensor = camera_high.data.output["rgb"]
        camera_high_data_tensor = camera_high_data_tensor.view(camera_high_data_tensor.shape[1:])
        camera_high_data = camera_high_data_tensor.cpu().numpy()

        observation = {}
        observation["task"] = self.scene_info["task"]
        observation["qpos"] = qpos
        observation["x_y_z_roll_pitch_yaw_gripper"] = x_y_z_roll_pitch_yaw_gripper
        observation["camera_wrist_right_data"] = camera_wrist_right_data
        observation["camera_high_data"] = camera_high_data
        return observation
        

    def capture_data(self):
        observation = self.get_current_observation()
        self.captured_data['/observations/x_y_z_roll_pitch_yaw_gripper'].append(observation["x_y_z_roll_pitch_yaw_gripper"])
        self.captured_data['/observations/qpos'].append(observation["qpos"])
        self.captured_data['/observations/images/camera_high'].append(observation["camera_high_data"])
        self.captured_data['/observations/images/camera_wrist_right'].append(observation["camera_wrist_right_data"])


    def create_hdf5_from_captured_data(self, save_directory, episode_index, sub_episode_last_step_index_list):
        camera_names = ["camera_high", "camera_wrist_right"]

        # compress images
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = self.captured_data[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                # 0.02 sec # cv2.imdecode(encoded_image, 1)
                _, encoded_image = cv2.imencode('.jpg', image, encode_param)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            self.captured_data[f'/observations/images/{cam_name}'] = compressed_list

        # pad so it has same length
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = self.captured_data[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            self.captured_data[f'/observations/images/{cam_name}'] = padded_compressed_image_list

        # HDF5
        n_steps = len(self.captured_data['/observations/qpos'])

        print(f"n_steps: {n_steps}")
        
        dataset_path = f"{save_directory}/episode_{episode_index}.hdf5"
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = True
            root.attrs['compress'] = True
            root.attrs['task'] = self.scene_info["task"]
            root.attrs['target_item_name'] = self.scene_info["target_item_name"]
            for item_name in self.scene_info["initial_item_positions"].keys():
                root.attrs[f"{item_name}_position"] = self.scene_info["initial_item_positions"][item_name]
            
            for item_name in self.scene_info["initial_item_orientations"].keys():
                root.attrs[f"{item_name}_orientation"] = self.scene_info["initial_item_orientations"][item_name]
            root.attrs['sub_episode_last_step_index_list'] = sub_episode_last_step_index_list

            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (n_steps, padded_size), dtype='uint8', chunks=(1, padded_size))

            _ = obs.create_dataset('qpos', (n_steps, 7))
            _ = obs.create_dataset('x_y_z_roll_pitch_yaw_gripper', (n_steps, 7))

            for name, array in self.captured_data.items():
                root[name][...] = array[:n_steps]

            _ = root.create_dataset('compress_len', (len(camera_names), n_steps))
            root['/compress_len'][...] = compressed_len[:, :n_steps]
