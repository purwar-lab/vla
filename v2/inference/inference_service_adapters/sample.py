import os
from pathlib import Path
import time

script_directory = Path(f"{os.path.dirname(__file__)}").parent

class InferenceService:
    def __init__(self, host, port=""):
        print("Connecting to inference server...")
        self.host = host
        self.port = port
        # Connect to inference server using host and port
        
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
        self.max_sub_steps = max_sub_steps
        self.step_index = -1
        # Reset inference server
        

    def predict(self, current_observation):
        self.step_index += 1

        # print(current_observation.keys())

        # Use current_observation to make inference call 
        
        actions = [0, 0, 0, 0, 0, 0, 0] # This is just to make the script work. You should get actual values from inference. 

        prediction = {}
        prediction["qpos"] = actions # If inference is joint angles (radians)

        # Alternatively, if the inference is eef position and orientation, then,
        # prediction["x_y_z_roll_pitch_yaw_gripper"] = actions
        # The system automatically applies qpos or x_y_z_roll_pitch_yaw_gripper, whichever is present.

        # Note: If inference gives delta values, then please add the delta to the current observation and then store it.

        prediction["terminate"] = False # If your inference contains a terminating state, then apply this accordingly.

        time.sleep(0.1)
        
        return prediction        
