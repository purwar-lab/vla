import os
script_directory = f"{os.path.dirname(__file__)}"
import sys
from pathlib import Path
sys.path.append(str(Path(script_directory).parent))
import importlib


import argparse

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="run_inference")
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run.")
parser.add_argument("--inference_service_name", type=str, default="", help="Inference service name")
parser.add_argument("--inference_server_host", type=str, default="localhost", help="Inference server host")
parser.add_argument("--inference_server_port", type=str, default="", help="Inference server port")
parser.add_argument("--hdf5_file_or_folder_path", type=str, default="", help="Path to hdf5 file or folder containing multiple files.")
parser.add_argument("--reporting_server_url", type=str, default="", help="Server where report is sent after comparison")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()



# import and instantiate inference service
inference_service_adapters_directory_name = "inference_service_adapters"
inference_service_adapters_directory_path = f"{script_directory}/{inference_service_adapters_directory_name}"
inference_adapters = {}
for filename in os.listdir(inference_service_adapters_directory_path):
    filepath = f"{inference_service_adapters_directory_path}/{filename}"
    if filename.endswith(".py") and os.path.isfile(filepath):
        module_name = filename[:-3]
        inference_adapters[module_name] = f"{os.path.basename(Path(script_directory))}.{inference_service_adapters_directory_name}.{module_name}"

if args_cli.inference_service_name == "": # Ask for user input 
    inference_adapters_names = list(inference_adapters.keys())

    choice = -1
    print("\n\nSelect Inference Service:")
    for i in range(len(inference_adapters_names)):
        print(f"    {i+1}: {inference_adapters_names[i]}")
    choice = int(input("\nEnter your choice: ")) - 1
    args_cli.inference_service_name = inference_adapters_names[choice]

inference_service_import = importlib.import_module(inference_adapters[args_cli.inference_service_name])
# Create inference_service instance here so that it can check the connection before starting anything else. Helps to debug bad connections.

num_inference_services = 4
inference_services = []
for i in range(num_inference_services):
    inference_service = inference_service_import.InferenceService(args_cli.inference_server_host, str(int(args_cli.inference_server_port)+i))
    inference_services.append(inference_service)


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""
import math
import shutil
import time
from threading import Thread
import requests
import sys
import h5py
import rerun as rr
from flask import Flask # Only for external trigger
from copy import deepcopy
import json
import re
from tqdm import tqdm
import torch

from utils import scene_util
from config import custom_items
from utils import custom_util

MIN_Z_HEIGHT_THRESHOLD = 0.08
MAX_Z_HEIGHT_THRESHOLD = 0.12 # incase the cuboid lands on its side

flask_app = Flask(__name__)
@flask_app.get("/")
def external_trigger():
    global external_trigger_flag
    if external_trigger_flag: # It's already processing the last request
        message = "Still processing the last request..."
        print(message)
        return message, 429
    external_trigger_flag = True
    message = "Process triggered."
    print(message)
    return message


def run_flask_app():
    global flask_app
    flask_app.run(debug=False, host='0.0.0.0', port=9000)


def inference_loop():
    global current_observation, make_inference_call_flag, prediction, current_inference_service
    while simulation_app.is_running():
        if make_inference_call_flag:
            prediction = current_inference_service.predict(current_observation)
            make_inference_call_flag = False
        else:
            time.sleep(0.01)


def run_simulator():
    global current_observation, make_inference_call_flag, prediction, external_trigger_flag, current_inference_service
    
    custom_scene = scene_util.CustomScene(args_cli)

    make_inference_call_flag = False
    current_observation = None

    inference_thread = Thread(target = inference_loop, daemon = True)
    inference_thread.start()

    send_report_flag = False
    if args_cli.reporting_server_url != "":
        send_report_flag = True
        external_trigger_flag = False
        flask_app_thread = Thread(target = run_flask_app, daemon = True)
        flask_app_thread.start()

    num_episodes = args_cli.num_episodes

    compare_with_real = False

    hdf5_file_paths = None
    run_dir = None

    if args_cli.hdf5_file_or_folder_path != "":
        hdf5_file_paths = []
        if os.path.isfile(args_cli.hdf5_file_or_folder_path) and args_cli.hdf5_file_or_folder_path.lower().endswith(".hdf5"):
            hdf5_file_paths.append(args_cli.hdf5_file_or_folder_path)
        else:
            file_names = os.listdir(args_cli.hdf5_file_or_folder_path)
            file_names = sorted(file_names, key=lambda x: int(re.search(r'\d+', x).group()))
            for file_name in file_names:
                if file_name.lower().endswith(".hdf5"):
                    hdf5_file_paths.append(f"{args_cli.hdf5_file_or_folder_path}/{file_name}")

        num_episodes = len(hdf5_file_paths)
        if num_episodes == 0:
            return
        # compare_with_real = True
    
    
    while simulation_app.is_running():
        if send_report_flag and not external_trigger_flag: # wait for trigger
            custom_scene.update_scene()
            time.sleep(1)
            continue

        if compare_with_real:
            hdf5_dir = os.path.dirname(hdf5_file_paths[0])
            run_dir = f"{hdf5_dir}/rerun_{round(time.time())}"
            os.makedirs(run_dir)
            inference_service.set_hdf5_file_paths_for_comparison(hdf5_file_paths, run_dir)

        episode_index = -1
        
        accuracy_json = {
            "total": 0,
            "placed_correct_item": 0,
            "placed_wrong_item": 0,
            "placed_correct_item_but_didnt_open_gripper": 0,
            "placed_wrong_item_but_didnt_open_gripper": 0,
            "didnt_place_any_item": 0,
            "received_termination_signal": 0
        }

        outcome_to_episode_json = {}
        for key in accuracy_json.keys():
            if key != "total":
                outcome_to_episode_json[key] = []

        while simulation_app.is_running() and episode_index < num_episodes:
            if episode_index == -1:
                custom_scene.initial_pause_for_lighting()
                episode_index = 0
            
            print("\n------------------------\n")
            max_steps = 80 #scene_util.MAX_STEPS
            if compare_with_real:
                hdf5_file_path = hdf5_file_paths[episode_index]
                hdf5_file_name = hdf5_file_path.split('/')[-1]
                print(f"Episode File: {hdf5_file_name}")
                scene_info = custom_scene.place_items_from_hdf5_file(hdf5_file_path)
            else:
                print(f"Episode Index: {episode_index}")
                scene_info = custom_scene.place_items_randomly(episode_index)
            
            print(f"Task: {scene_info['task']}")
            for model_index, inference_service in enumerate(inference_services):
                current_inference_service = inference_service
                current_inference_service.reset(episode_index, max_steps)
                
                terminated = False

                for _ in tqdm(range(max_steps), desc=f"Model {model_index}: Step"):
                    current_observation = custom_scene.get_current_observation()
                    make_inference_call_flag = True
                    while make_inference_call_flag:
                        custom_scene.update_scene() # wait for inference without slowing down simulation

                    if prediction.get("qpos_list", None) != None: # In case of multiple steps/action chunks
                        for qpos in prediction["qpos_list"]:
                            custom_scene.apply_qpos_to_robot(qpos)
                    elif prediction.get("x_y_z_roll_pitch_yaw_gripper_list", None) != None: # In case of multiple steps/action chunks
                        for x_y_z_roll_pitch_yaw_gripper in prediction["x_y_z_roll_pitch_yaw_gripper_list"]:
                            custom_scene.move_ee_to_pos(x_y_z_roll_pitch_yaw_gripper, world=False)
                    elif prediction.get("qpos", None) != None:
                        custom_scene.apply_qpos_to_robot(prediction["qpos"])
                    elif prediction.get("x_y_z_roll_pitch_yaw_gripper", None) != None:
                        custom_scene.move_ee_to_pos(prediction["x_y_z_roll_pitch_yaw_gripper"], world=False)
                    if prediction.get("terminate", False):
                        terminated = True
                        break
            
            
            # evaluate
            accuracy_json["total"] += 1

            current_observation = custom_scene.get_current_observation()

            [platform_x, platform_y, _, _, _, _] = custom_scene.get_current_x_y_z_roll_pitch_yaw_of_item(scene_util.PLATFORM)

            is_gripper_open = bool(current_observation["x_y_z_roll_pitch_yaw_gripper"][6])

            placed_some_item = False
            for item_name in scene_util.ITEM_NAMES:
                initial_item_z = scene_info["initial_item_positions"][item_name][2]
                [current_item_x, current_item_y, current_item_z] = custom_scene.get_current_x_y_z_roll_pitch_yaw_of_item(item_name)[:3]
                z_height = current_item_z - initial_item_z
                delta_x = abs(current_item_x - platform_x)
                delta_y = abs(current_item_y - platform_y)
                if MIN_Z_HEIGHT_THRESHOLD <= z_height <= MAX_Z_HEIGHT_THRESHOLD and delta_x <= 0.095 and delta_y <= 0.15:
                    key = ""
                    placed_some_item = True
                    if item_name == scene_info["target_item_name"]:
                        if is_gripper_open:
                            key = "placed_correct_item"
                        else:
                           key = "placed_correct_item_but_didnt_open_gripper"
                    else:
                        if is_gripper_open:
                           key = "placed_wrong_item"
                        else:
                           key = "placed_wrong_item_but_didnt_open_gripper"
                    accuracy_json[key] += 1
                    if compare_with_real:
                        outcome_to_episode_json[key].append(hdf5_file_name)
                    break
            
            if not placed_some_item:
                key = "didnt_place_any_item"
                accuracy_json[key] += 1
                if compare_with_real:
                    outcome_to_episode_json[key].append(hdf5_file_name)
            
            if terminated:
                key = "received_termination_signal"
                accuracy_json[key] += 1
                if compare_with_real:
                    outcome_to_episode_json[key].append(hdf5_file_name)

            print()
            print(f"total: {accuracy_json['total']}")
            for key in outcome_to_episode_json:
                if key == "total":
                    continue
                percentage = 100 * accuracy_json[key] / accuracy_json["total"]
                print(f"{key}: {round(percentage, 2)}% ({accuracy_json[key]} / {accuracy_json['total']})")

            
            if compare_with_real:
                with open(f"{run_dir}/accuracy_json.json", "w") as fobj: # Save it after every episode
                    json.dump(accuracy_json, fobj)
                
                with open(f"{run_dir}/outcome_to_episode_json.json", "w") as fobj: # Save it after every episode
                    json.dump(outcome_to_episode_json, fobj)

            episode_index += 1
                
        if send_report_flag:
            print("\nSending data to reporting server...")
            try:
                files = {
                    "rmse": (None, str(accuracy_json["placed_correct_item"] / accuracy_json["total"])) # (filename, content); filename=None means it's just a field. This is a hack to send the request as a multipart form request
                } # The field says rmse, but that hasn't beermsen finalized yet. Sending accuracy instead.
                response = requests.post(args_cli.reporting_server_url, files = files)
                response.raise_for_status() # raise exception in case of error status code
                print(f"Successfully sent report to reporting server.")
            except Exception as e:
                print(f"Couldn't send report to reporting server!!!")
            external_trigger_flag = False
        else:
            print("\n\nDone! Press CTRL+C to exit...\n")
            break




if __name__ == "__main__":
    # run the main function
    run_simulator()
    # close sim app
    simulation_app.close()