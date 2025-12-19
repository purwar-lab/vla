"""
.. code-block:: bash

    # Usage
    source env/bin/activate
    ./IsaacLab/isaaclab.sh -p custom_scripts/data_creation/create_data.py [--num_episodes 10] --enable_cameras

"""

import os
script_directory = f"{os.path.dirname(__file__)}"
import sys
from pathlib import Path
sys.path.append(str(Path(script_directory).parent))

import argparse

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="create_data")
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--first_episode_index", type=int, default=0, help="First episode index.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")
parser.add_argument("--resume", action="store_true", help="Resume data creation.")
parser.add_argument("--s3_dir", type=str, default="", help="s3 directory, where to upload files.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import math
import shutil
import time
import random
import re
from copy import deepcopy
import subprocess
from threading import Thread

from utils import scene_util
from config import custom_items
from utils import custom_util

def get_s3_file_list(s3_save_directory):
    output = subprocess.run(f"aws s3 ls {s3_save_directory}", shell=True, text=True, capture_output=True)
    if len(output.stderr) > 0:
        raise Exception(output.stderr)
    file_list = []
    for x in output.stdout.split("\n"):
        file_name = x.split(" ")[-1]
        if file_name.endswith(".hdf5"):
            file_list.append(file_name)
    return file_list


def s3_uploader(s3_save_directory, save_directory, first_episode_index, last_episode_index):
    print(s3_save_directory)
    all_file_names = [f"episode_{i}.hdf5" for i in range(first_episode_index, last_episode_index+1)]
    uploaded_file_names = get_s3_file_list(s3_save_directory)
    pending_file_names = list(set(all_file_names) - set(uploaded_file_names))
    while len(pending_file_names) > 0:
        new_file_names = list(set(pending_file_names).intersection(set(os.listdir(save_directory))))
        new_file_names = sorted(new_file_names, key=lambda x: int(re.search(r'\d+', x).group()))
        for file_name in new_file_names:
            try:
                print(f"Uploading {file_name}...")
                output = subprocess.run(f"aws s3 cp {save_directory}/{file_name} {s3_save_directory}", shell=True, text=True, capture_output=True)
                if len(output.stderr) > 0:
                    raise Exception(output.stderr)
                os.remove(f"{save_directory}/{file_name}")
                pending_file_names.remove(file_name)
                print(f"Uploaded {file_name}")
            except:
                pass
        time.sleep(1)


def clear_s3_files(s3_save_directory):
    output = subprocess.run(f"aws s3 rm --recursive {s3_save_directory}", shell=True, text=True, capture_output=True)
    if len(output.stderr) > 0:
        raise Exception(output.stderr)


save_directory = f"{script_directory}/saved_data"

if not args_cli.resume and os.path.isdir(save_directory):
    shutil.rmtree(save_directory)

os.makedirs(save_directory, exist_ok=True)

s3_dir = args_cli.s3_dir
if s3_dir.endswith("/"):
    s3_dir = s3_dir[:-1]
s3_save_directory = f"{s3_dir}/saved_data/"

if not args_cli.resume and s3_dir != "":
    clear_s3_files(s3_save_directory)

def run_simulator():
    """Runs the simulation loop."""
    custom_scene = scene_util.CustomScene(args_cli)

    num_episodes = args_cli.num_episodes
    first_episode_index = args_cli.first_episode_index
    last_episode_index = first_episode_index + num_episodes - 1
    existing_episode_index_dict = {}

    if args_cli.resume:
        file_names = []
        for file_name in os.listdir(save_directory):
            if file_name.endswith(".hdf5"):
                file_names.append(file_name)
        
        if s3_dir != "":
            file_names += get_s3_file_list(s3_save_directory)

        for file_name in file_names:
            if file_name.lower().endswith(".hdf5"):
                episode_index = int(file_name.replace("episode_", "").replace(".hdf5", ""))
                existing_episode_index_dict[episode_index] = True
    
    if s3_dir != "":
        s3_uploader_thread = Thread(target=s3_uploader, args=(s3_save_directory, save_directory, first_episode_index, last_episode_index), daemon=True)
        s3_uploader_thread.start()
    
    episode_index = -1 # initialization
    
    # Simulation loop
    start_time = time.time()
    accepted_episodes_count = 0
    discarded_episodes_count = 0
    
    final_delta_radians = math.radians(10)
    
    while simulation_app.is_running() and episode_index <= last_episode_index:
        if episode_index == -1: # Initialization run for lighting and stuff
            custom_scene.initial_pause_for_lighting()
            episode_index = first_episode_index
            continue
        
        if existing_episode_index_dict.get(episode_index, False): # Episode already exists
            episode_index += 1
            continue

        episode_start_time = time.time()
        # reset joint state
        print("\n------------------------\n")
        print(f"Episode Index: {episode_index}")

        scene_info = custom_scene.place_items_randomly(episode_index)
        custom_scene.capture_data()
        sub_episode_last_step_index_list = []

        task = scene_info["task"]
        print(task)

        target_item_name = scene_info["target_item_name"]

        current_x_y_z_roll_pitch_yaw_gripper = deepcopy(scene_info["initial_x_y_z_roll_pitch_yaw_gripper"])

        target_x_y_z_roll_pitch_yaw_gripper_list = []

        # First move to touch item
        target_x_y_z_roll_pitch_yaw_gripper = deepcopy(scene_info["grab_x_y_z_roll_pitch_yaw_gripper"])
        target_x_y_z_roll_pitch_yaw_gripper[0] -= scene_util.GRIPPER_BUFFER + 0.01
        target_x_y_z_roll_pitch_yaw_gripper_list.append(target_x_y_z_roll_pitch_yaw_gripper)

        # Move to place item inside gripper
        target_x_y_z_roll_pitch_yaw_gripper = deepcopy(scene_info["grab_x_y_z_roll_pitch_yaw_gripper"])
        target_x_y_z_roll_pitch_yaw_gripper[0] -= 0.01
        target_x_y_z_roll_pitch_yaw_gripper_list.append(target_x_y_z_roll_pitch_yaw_gripper)

        # Lift up item
        target_x_y_z_roll_pitch_yaw_gripper = deepcopy(scene_info["place_x_y_z_roll_pitch_yaw_gripper"])
        target_x_y_z_roll_pitch_yaw_gripper[0] -= 0.04 + scene_util.GRIPPER_BUFFER
        target_x_y_z_roll_pitch_yaw_gripper_list.append(target_x_y_z_roll_pitch_yaw_gripper)

        # Move to place item above platform
        target_x_y_z_roll_pitch_yaw_gripper = deepcopy(scene_info["place_x_y_z_roll_pitch_yaw_gripper"])
        target_x_y_z_roll_pitch_yaw_gripper[0] -= 0.02
        target_x_y_z_roll_pitch_yaw_gripper_list.append(target_x_y_z_roll_pitch_yaw_gripper)

        for path_index, target_x_y_z_roll_pitch_yaw_gripper in enumerate(target_x_y_z_roll_pitch_yaw_gripper_list):
            direction_signs = []
            for i in range(6):
                if target_x_y_z_roll_pitch_yaw_gripper[i] < current_x_y_z_roll_pitch_yaw_gripper[i]:
                    direction_signs.append(-1)
                else:
                    direction_signs.append(1)
            
            while True:
                no_change_count = 0

                # position along straight line
                random_delta = random.uniform(0.002, 0.006)
                current_distance_to_target = custom_util.get_distance_between_two_points(current_x_y_z_roll_pitch_yaw_gripper[:3], target_x_y_z_roll_pitch_yaw_gripper[:3])
                random_delta = min(random_delta, current_distance_to_target)
                if random_delta < 0.001:
                    no_change_count += 3
                else:
                    distance_ratio = random_delta / current_distance_to_target
                    for i in range(3):
                        current_x_y_z_roll_pitch_yaw_gripper[i] += distance_ratio * (target_x_y_z_roll_pitch_yaw_gripper[i] - current_x_y_z_roll_pitch_yaw_gripper[i])

                # orientation
                for i in range(3,6):
                    random_delta = random.uniform(0.002, 0.006)
                    current_delta = abs(target_x_y_z_roll_pitch_yaw_gripper[i] - current_x_y_z_roll_pitch_yaw_gripper[i])
                    random_delta = min(random_delta, current_delta)
                    if random_delta <= 0.001:
                        no_change_count += 1
                    else:
                        current_x_y_z_roll_pitch_yaw_gripper[i] += random_delta * direction_signs[i]
                
                if no_change_count == 6:
                    break

                custom_scene.move_ee_to_pos(current_x_y_z_roll_pitch_yaw_gripper, world=True) # not setting capture_data_flag=True because we don't want to capture intermediate steps in this case
                custom_scene.capture_data()
            
            if path_index == 1:
                # Close gripper
                custom_scene.operate_gripper(scene_util.GRIPPER_ACTIONS["close"])
                custom_scene.capture_data()
                current_x_y_z_roll_pitch_yaw_gripper[6] = scene_util.GRIPPER_ACTIONS["close"]
            
            if path_index == 3:
                # Open gripper
                custom_scene.operate_gripper(scene_util.GRIPPER_ACTIONS["open"])
                custom_scene.capture_data()
                current_x_y_z_roll_pitch_yaw_gripper[6] = scene_util.GRIPPER_ACTIONS["open"]

            sub_episode_last_step_index_list.append(len(custom_scene.captured_data['/observations/x_y_z_roll_pitch_yaw_gripper'])-1)
        
        # Wait for some time before checking
        for _ in range(10):
            custom_scene.update_scene()

        current_target_item_x_y_z_roll_pitch_yaw = custom_scene.get_current_x_y_z_roll_pitch_yaw_of_item(target_item_name)
        
        # Check if episode was successful
        initial_target_item_x_y_z = scene_info["initial_item_positions"][target_item_name]

        z_height = current_target_item_x_y_z_roll_pitch_yaw[2] - initial_target_item_x_y_z[2]
        roll = current_target_item_x_y_z_roll_pitch_yaw[3]
        pitch = current_target_item_x_y_z_roll_pitch_yaw[4]
        
        print(f"z_height: {z_height}")
        print(f"roll: {roll}")
        print(f"pitch: {pitch}")
        
        if z_height >= 0.03 and abs(roll) < final_delta_radians and abs(pitch) < final_delta_radians:
            accepted_episodes_count += 1
            # create_hdf5_from_captured_data(data_dict, episode_index, task, pick_up_item_name, place_on_item_name, scene_info["initial_item_positions"], scene_info["initial_item_orientations"])
            custom_scene.create_hdf5_from_captured_data(save_directory, episode_index, sub_episode_last_step_index_list)
            start_x = 0
            for i, x in enumerate(sub_episode_last_step_index_list):
                print(f"Part {i+1} n_steps: {x - start_x + 1}")
                start_x = x + 1
            time_taken_for_episode = round(time.time() - episode_start_time, 1)
            print(f"Time taken: {time_taken_for_episode} seconds")
            episode_index += 1
        else:
            print("Episode discarded!!!")
            discarded_episodes_count += 1
    
    if s3_dir != "":
        s3_uploader_thread.join()
    
    total_episodes_count = accepted_episodes_count + discarded_episodes_count
    if total_episodes_count > 0:
        print(f"\n\nTotal time taken: {round((time.time() - start_time)/3600, 2)} hours")
        print(f"\nTotal number of generated episodes: {total_episodes_count}")
        print(f"Number of accepted episodes: {accepted_episodes_count}")
        print(f"Episode acceptance percentage: {round(100 * accepted_episodes_count / (total_episodes_count), 2)} %")
    print("\n\nDone! Press CTRL+C to exit...\n")

if __name__ == "__main__":
    # run the main function
    run_simulator()
    # close sim app
    simulation_app.close()