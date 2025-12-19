import h5py
import rerun as rr
import time
import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import os
from pathlib import Path

script_directory = Path(f"{os.path.dirname(__file__)}").parent

class InferenceService:
    def __init__(self, host, port=""):
        print("Connecting to inference server...")
        self.host = host
        self.port = port
        self.pi_zero_client = _websocket_client_policy.WebsocketClientPolicy(host=self.host, port=self.port)
        
        self.hdf5_file_paths = None
        self.run_dir = None
        self.current_hdf5_obj = None
        self.episode_index = -1
        self.step_index = -1
        self.max_steps = -1
        self.reset()
    
    def set_hdf5_file_paths_for_comparison(self, hdf5_file_paths, run_dir):
        self.hdf5_file_paths = hdf5_file_paths
        self.run_dir = run_dir
    
    def reset(self, episode_index=-1, max_steps=-1, max_sub_steps=-1):
        self.episode_index = episode_index
        self.max_steps = max_steps
        self.max_sub_steps = max_sub_steps
        self.step_index = -1
        if self.hdf5_file_paths != None:
            if self.current_hdf5_obj != None:
                self.current_hdf5_obj.close()
            self.current_hdf5_obj = h5py.File(self.hdf5_file_paths[episode_index], "r")
            task = self.current_hdf5_obj.attrs["task"]
            self.episode_file_name = self.hdf5_file_paths[episode_index].split("/")[-1]
            rr_episode_name = f"{self.episode_file_name[:-5]} ({task})"
            rr.init(rr_episode_name, spawn=False)
        

    def predict(self, current_observation):
        self.step_index += 1

        data = {
            "state": [0] * 14,
            "images": {},
            "prompt": current_observation["task"]
        }
        data["state"][:7] = current_observation["qpos"]

        data["images"]["cam_high"] = image_tools.resize_with_pad(current_observation["camera_high_data"], 224, 224)
        data["images"]["cam_right_wrist"] = image_tools.resize_with_pad(current_observation["camera_wrist_right_data"], 224, 224)
        data["images"]["cam_left_wrist"] = np.zeros((224, 224, 3), dtype=np.uint8)
        data["images"]["cam_low"] = np.zeros((224, 224, 3), dtype=np.uint8)

        for camera_name in data["images"].keys():
            camera_data = data["images"][camera_name]
            camera_data_after_reordering = np.asarray([camera_data[:,:,0], camera_data[:,:,1], camera_data[:,:,2]]) # For some reason pi_zero wants (3, 224, 224)
            data["images"][camera_name] = camera_data_after_reordering
        
        actions = self.pi_zero_client.infer(data)["actions"][:, :7].tolist()

        prediction = {}
        prediction["qpos_list"] = actions[0:self.max_sub_steps]
        prediction["terminate"] = False
        if self.step_index == self.max_steps-1:
            prediction["terminate"] = True
        
        
        if self.current_hdf5_obj != None: # record rerun
            n_real_steps = len(self.current_hdf5_obj["observations"]["qpos"])
            rr.set_time("step_index", sequence=self.step_index)
            
            inference_camera_high_image = current_observation["camera_high_data"]
            inference_camera_wrist_right_image = current_observation["camera_wrist_right_data"]

            self._log_rr(inference_camera_high_image, inference_camera_wrist_right_image, prediction["qpos_list"][-1])

            if self.step_index < n_real_steps-1:
                image_bytes = self.current_hdf5_obj["observations"]["images"]["camera_high"][self.step_index]
                real_camera_high_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                image_bytes = self.current_hdf5_obj["observations"]["images"]["camera_wrist_right"][self.step_index]
                real_camera_wrist_right_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                real_qpos = self.current_hdf5_obj["observations"]["qpos"][self.step_index+1]
                self._log_rr(real_camera_high_image, real_camera_wrist_right_image, real_qpos, real_flag=True)
                
        
            if prediction["terminate"] or self.step_index == self.max_steps-1:
                # Record any remaining real steps
                for i in range(self.step_index+1, n_real_steps-1):
                    rr.set_time("step_index", sequence=i)
                    image_bytes = self.current_hdf5_obj["observations"]["images"]["camera_high"][i]
                    real_camera_high_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    image_bytes = self.current_hdf5_obj["observations"]["images"]["camera_wrist_right"][i]
                    real_camera_wrist_right_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    real_qpos = self.current_hdf5_obj["observations"]["qpos"][i+1]
                    self._log_rr(real_camera_high_image, real_camera_wrist_right_image, real_qpos, real_flag=True)
                    
                rr.save(f"{self.run_dir}/{self.episode_file_name[:-5]}.rrd")
                self.current_hdf5_obj.close()
        
        return prediction        

    def _log_rr(self, camera_high_image, camera_wrist_right_image, actions, real_flag=False):
        prefix = "inference_"
        if real_flag:
            prefix = "real_"
        rr.log(f"{prefix}camera_high_image", rr.Image(camera_high_image))
        rr.log(f"{prefix}camera_wrist_right_image", rr.Image(camera_wrist_right_image))

        rr.log(f"{prefix}actions/waist", rr.Scalars(actions[0]))
        rr.log(f"{prefix}actions/shoulder", rr.Scalars(actions[1]))
        rr.log(f"{prefix}actions/elbow", rr.Scalars(actions[2]))
        rr.log(f"{prefix}actions/forearm_roll", rr.Scalars(actions[3]))
        rr.log(f"{prefix}actions/wrist_angle", rr.Scalars(actions[4]))
        rr.log(f"{prefix}actions/wrist_rotate", rr.Scalars(actions[5]))
        rr.log(f"{prefix}actions/gripper", rr.Scalars(actions[6]))
