from gr00t.eval.robot import RobotInferenceClient
import numpy as np
import os
from pathlib import Path
import time

script_directory = Path(f"{os.path.dirname(__file__)}").parent


class InferenceService:
    STEPS_PER_INFER = 16
    actions: np.ndarray  # shape: (16, x); x = DoF (7 for SO-100)

    def __init__(self, host, port=""):
        print("Connecting to inference server...")
        self.host = host
        self.port = port
        # Connect to inference server using host and port
        self.client = RobotInferenceClient(host=host, port=port)
        # self.actions = np.ndarray((16, 0))

        self.episode_index = -1
        self.step_index = -1
        self.max_steps = -1
        self.reset()

    def set_hdf5_file_paths_for_comparison(self, hdf5_file_paths, run_dir):
        # This method is required for comparing results through rerun.
        # Even if you don't want to implement rerun, please keep this method to prevent the calling script from breaking.
        self.hdf5_file_paths = hdf5_file_paths
        self.run_dir = run_dir

    def reset(self, episode_index=-1, max_steps=-1, max_sub_steps=-1):
        self.episode_index = episode_index
        self.max_steps = max_steps
        self.step_index = -1
        # Reset inference server

    def predict(self, current_observation):
        self.step_index += 1
        action_index = self.step_index % self.STEPS_PER_INFER
        # print(f"Action index: {action_index}")
        if action_index == 0:
            img_front = np.array(current_observation["camera_high_data"])
            img_wrist = np.array(current_observation["camera_wrist_right_data"])
            state = np.array(current_observation["qpos"])

            # Camera changed to be 300x300
            # img_front = cv2.resize(
            #     img_front, (300, 300), interpolation=cv2.INTER_AREA
            # )
            # img_wrist = cv2.resize(
            #     img_wrist, (300, 300), interpolation=cv2.INTER_AREA
            # )

            obs = {
                "video.front": img_front.reshape(1, 300, 300, 3),
                "video.wrist": img_wrist.reshape(1, 300, 300, 3),
                "state.single_arm": state[:6].reshape(1, 6).astype(np.float64),
                "state.gripper": state[6].reshape(1, 1).astype(np.float64),
                "annotation.human.action.task_description": [
                    current_observation["task"]
                ],
            }
            resp = self.client.get_action(obs)
            self.actions = np.concatenate(
                (resp["action.single_arm"], resp["action.gripper"].reshape(-1, 1)),
                axis=1,
            )
            # print(f"New actions received. Shape: {self.actions.shape}")

        prediction = {}
        prediction["qpos"] = self.actions[action_index]
        prediction["terminate"] = False
        time.sleep(0.1)

        return prediction
