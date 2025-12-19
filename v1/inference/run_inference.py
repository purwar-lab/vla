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
parser.add_argument(
    "--num_episodes", type=int, default=0, help="Number of episodes to run."
)  # default is set to 1 later on in the code
parser.add_argument(
    "--inference_service_name", type=str, default="", help="Inference service name"
)
parser.add_argument(
    "--inference_server_host",
    type=str,
    default="localhost",
    help="Inference server host",
)
parser.add_argument(
    "--inference_server_port", type=str, default="", help="Inference server port"
)
parser.add_argument(
    "--hdf5_file_or_folder_path",
    type=str,
    default="",
    help="Path to hdf5 file or folder containing multiple files.",
)
parser.add_argument(
    "--reporting_server_url",
    type=str,
    default="",
    help="Server where report is sent after comparison",
)
parser.add_argument(
    "--max_steps", type=int, default=-1, help="Maximum number of steps for inference."
)
parser.add_argument(
    "--max_sub_steps",
    type=int,
    default=-1,
    help="Maximum number of sub-steps for inference.",
)
parser.add_argument(
    "--result_folder_path",
    type=str,
    default="",
    help="Path to the folder for storing results.",
)
parser.add_argument("--resume", action="store_true", help="Resume evaluation.")
parser.add_argument(
    "--s3_dir", type=str, default="", help="s3 directory, where to upload files."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# import and instantiate inference service
inference_service_adapters_directory_name = "inference_service_adapters"
inference_service_adapters_directory_path = (
    f"{script_directory}/{inference_service_adapters_directory_name}"
)
inference_adapters = {}
for filename in os.listdir(inference_service_adapters_directory_path):
    filepath = f"{inference_service_adapters_directory_path}/{filename}"
    if filename.endswith(".py") and os.path.isfile(filepath):
        module_name = filename[:-3]
        inference_adapters[module_name] = (
            f"{os.path.basename(Path(script_directory))}.{inference_service_adapters_directory_name}.{module_name}"
        )

if args_cli.inference_service_name == "":  # Ask for user input
    inference_adapters_names = list(inference_adapters.keys())

    choice = -1
    print("\n\nSelect Inference Service:")
    for i in range(len(inference_adapters_names)):
        print(f"    {i + 1}: {inference_adapters_names[i]}")
    choice = int(input("\nEnter your choice: ")) - 1
    args_cli.inference_service_name = inference_adapters_names[choice]

inference_service_import = importlib.import_module(
    inference_adapters[args_cli.inference_service_name]
)
# Create inference_service instance here so that it can check the connection before starting anything else. Helps to debug bad connections.
inference_service = inference_service_import.InferenceService(
    args_cli.inference_server_host, args_cli.inference_server_port
)


"""Rest everything follows."""
import math
import shutil
import time
from threading import Thread
import requests
import sys
import h5py
import rerun as rr
from copy import deepcopy
import json
import re
from tqdm import tqdm
import torch
import subprocess

from utils import scene_util
from config import custom_items
from utils import custom_util

MIN_Z_HEIGHT_THRESHOLD = 0.08
MAX_Z_HEIGHT_THRESHOLD = 0.12  # incase the cuboid lands on its side


def inference_loop():
    global current_observation, make_inference_call_flag, prediction
    while simulation_app.is_running():
        if make_inference_call_flag:
            prediction = inference_service.predict(current_observation)
            make_inference_call_flag = False
        else:
            time.sleep(0.01)


def download_from_s3(s3_file_path, local_file_path):
    print(f"Downloading {s3_file_path}...")
    output = subprocess.run(
        f"aws s3 cp {s3_file_path} {local_file_path}",
        shell=True,
        text=True,
        capture_output=True,
    )
    if len(output.stderr) > 0:
        raise Exception(output.stderr)
    print("Downloaded.")


def upload_to_s3(local_file_path, s3_file_path, delete_local=False):
    print(f"Uploading {local_file_path}...")
    output = subprocess.run(
        f"aws s3 cp {local_file_path} {s3_file_path}",
        shell=True,
        text=True,
        capture_output=True,
    )
    if len(output.stderr) > 0:
        raise Exception(output.stderr)
    print("Uploaded.")
    if delete_local:
        os.remove(local_file_path)


def run_simulator():
    global current_observation, make_inference_call_flag, prediction

    custom_scene = scene_util.CustomScene(args_cli)

    make_inference_call_flag = False
    current_observation = None

    inference_thread = Thread(target=inference_loop, daemon=True)
    inference_thread.start()

    send_report_flag = False
    if args_cli.reporting_server_url != "":
        send_report_flag = True

    num_episodes = args_cli.num_episodes

    compare_with_real = False

    hdf5_file_paths = None
    run_dir = args_cli.result_folder_path
    s3_dir = args_cli.s3_dir
    if s3_dir.endswith("/"):
        s3_dir = s3_dir[:-1]

    if args_cli.hdf5_file_or_folder_path != "":
        hdf5_file_paths = []
        if os.path.isfile(
            args_cli.hdf5_file_or_folder_path
        ) and args_cli.hdf5_file_or_folder_path.lower().endswith(".hdf5"):
            hdf5_file_paths.append(args_cli.hdf5_file_or_folder_path)
        else:
            file_names = os.listdir(args_cli.hdf5_file_or_folder_path)
            file_names = sorted(
                file_names, key=lambda x: int(re.search(r"\d+", x).group())
            )
            for file_name in file_names:
                if file_name.lower().endswith(".hdf5"):
                    hdf5_file_paths.append(
                        f"{args_cli.hdf5_file_or_folder_path}/{file_name}"
                    )
        if num_episodes > 0:
            hdf5_file_paths = hdf5_file_paths[:num_episodes]
        num_episodes = len(
            hdf5_file_paths
        )  # incase the folder doesn't have that many files

        compare_with_real = True

    elif num_episodes <= 0:
        num_episodes = 1  # default

    while (
        simulation_app.is_running()
    ):  # This loop is not required anymore, but keeping it to not lengthen the commit
        if compare_with_real:
            if run_dir == "":
                hdf5_dir = os.path.dirname(hdf5_file_paths[0])
                run_dir = f"{hdf5_dir}/rerun_{round(time.time())}"

            if not args_cli.resume and os.path.isdir(run_dir):
                shutil.rmtree(run_dir)
            os.makedirs(run_dir, exist_ok=True)
            inference_service.set_hdf5_file_paths_for_comparison(
                hdf5_file_paths, run_dir
            )

        episode_index = -1

        accuracy_json = {
            "total": 0,
            "placed_correct_item": 0,
            "placed_wrong_item": 0,
            "placed_correct_item_but_didnt_open_gripper": 0,
            "placed_wrong_item_but_didnt_open_gripper": 0,
            "didnt_place_any_item": 0,
            "received_termination_signal": 0,
        }

        outcome_to_episode_json = {}
        for key in accuracy_json.keys():
            if key != "total":
                outcome_to_episode_json[key] = []

        result_json_file_name = "result.json"
        result_json_file_path = f"{run_dir}/{result_json_file_name}"

        evaluated_episode_file_name_dict = {}
        accuracy_metric = 0

        if compare_with_real and args_cli.resume:
            if s3_dir != "":
                try:
                    download_from_s3(
                        f"{s3_dir}/{result_json_file_name}", result_json_file_path
                    )
                except:
                    pass

            try:
                with open(result_json_file_path) as fobj:
                    json_data = json.load(fobj)
                    accuracy_json = json_data["accuracy_json"]
                    outcome_to_episode_json = json_data["outcome_to_episode_json"]
                    for key in outcome_to_episode_json:
                        for file_name in outcome_to_episode_json[key]:
                            evaluated_episode_file_name_dict[file_name] = True
            except:
                pass

        while simulation_app.is_running() and episode_index < num_episodes:
            if episode_index == -1:
                custom_scene.initial_pause_for_lighting()
                episode_index = 0

            print("\n------------------------\n")
            max_steps = args_cli.max_steps
            if max_steps == -1:
                max_steps = scene_util.MAX_STEPS

            max_sub_steps = args_cli.max_sub_steps
            if max_sub_steps == -1:
                max_sub_steps = scene_util.MAX_SUB_STEPS

            if compare_with_real:
                hdf5_file_path = hdf5_file_paths[episode_index]
                hdf5_file_name = hdf5_file_path.split("/")[-1]
                if evaluated_episode_file_name_dict.get(hdf5_file_name, False):
                    episode_index += 1
                    continue
                print(f"Episode File: {hdf5_file_name}")
                scene_info = custom_scene.place_items_from_hdf5_file(hdf5_file_path)
            else:
                print(f"Episode Index: {episode_index}")
                scene_info = custom_scene.place_items_randomly(episode_index)

            print(f"Task: {scene_info['task']}")
            inference_service.reset(episode_index, max_steps, max_sub_steps)

            terminated = False

            for _ in tqdm(range(max_steps), desc="Step"):
                current_observation = custom_scene.get_current_observation()
                make_inference_call_flag = True
                while make_inference_call_flag:
                    custom_scene.update_scene()  # wait for inference without slowing down simulation

                if (
                    prediction.get("qpos_list", None) is not None
                ):  # In case of multiple steps/action chunks
                    for qpos in prediction["qpos_list"]:
                        custom_scene.apply_qpos_to_robot(qpos)
                elif (
                    prediction.get("x_y_z_roll_pitch_yaw_gripper_list", None)
                    is not None
                ):  # In case of multiple steps/action chunks
                    for x_y_z_roll_pitch_yaw_gripper in prediction[
                        "x_y_z_roll_pitch_yaw_gripper_list"
                    ]:
                        custom_scene.move_ee_to_pos(
                            x_y_z_roll_pitch_yaw_gripper, world=False
                        )
                elif prediction.get("qpos", None) is not None:
                    custom_scene.apply_qpos_to_robot(prediction["qpos"])
                elif prediction.get("x_y_z_roll_pitch_yaw_gripper", None) is not None:
                    custom_scene.move_ee_to_pos(
                        prediction["x_y_z_roll_pitch_yaw_gripper"], world=False
                    )
                if prediction.get("terminate", False):
                    terminated = True
                    break

            # evaluate
            accuracy_json["total"] += 1

            current_observation = custom_scene.get_current_observation()

            [platform_x, platform_y, _, _, _, _] = (
                custom_scene.get_current_x_y_z_roll_pitch_yaw_of_item(
                    scene_util.PLATFORM
                )
            )

            is_gripper_open = bool(
                current_observation["x_y_z_roll_pitch_yaw_gripper"][6]
            )

            placed_some_item = False
            for item_name in scene_util.ITEM_NAMES:
                initial_item_z = scene_info["initial_item_positions"][item_name][2]
                [current_item_x, current_item_y, current_item_z] = (
                    custom_scene.get_current_x_y_z_roll_pitch_yaw_of_item(item_name)[:3]
                )
                z_height = current_item_z - initial_item_z
                delta_x = abs(current_item_x - platform_x)
                delta_y = abs(current_item_y - platform_y)
                if (
                    MIN_Z_HEIGHT_THRESHOLD <= z_height <= MAX_Z_HEIGHT_THRESHOLD
                    and delta_x <= 0.095
                    and delta_y <= 0.15
                ):
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
                print(
                    f"{key}: {round(percentage, 2)}% ({accuracy_json[key]} / {accuracy_json['total']})"
                )

                accuracy_metric = (
                    accuracy_json["placed_correct_item"] / accuracy_json["total"]
                )

            if compare_with_real:
                json_data = {
                    "accuracy_metric": accuracy_metric,
                    "accuracy_json": accuracy_json,
                    "outcome_to_episode_json": outcome_to_episode_json,
                }
                with open(
                    result_json_file_path, "w"
                ) as fobj:  # Save it after every episode
                    json.dump(json_data, fobj, indent=4)

                if s3_dir != "":
                    rerun_file_path = f"{run_dir}/{hdf5_file_name[:-5]}.rrd"
                    if os.path.isfile(rerun_file_path):
                        upload_to_s3(rerun_file_path, f"{s3_dir}/", delete_local=True)
                    upload_to_s3(result_json_file_path, f"{s3_dir}/")

            episode_index += 1

        if send_report_flag:
            print("\nSending data to reporting server...")
            try:
                files = {
                    "rmse": (
                        None,
                        str(accuracy_metric),
                    )  # (filename, content); filename=None means it's just a field. This is a hack to send the request as a multipart form request
                }  # The field says rmse, but that hasn't been finalized yet. Sending accuracy instead.
                response = requests.post(args_cli.reporting_server_url, files=files)
                response.raise_for_status()  # raise exception in case of error status code
                print(f"Successfully sent report to reporting server.")
            except Exception as e:
                print(f"Couldn't send report to reporting server!!!")
        print("\n\nDone! Press CTRL+C to exit...\n")
        break


if __name__ == "__main__":
    # run the main function
    run_simulator()
    # close sim app
    simulation_app.close()
