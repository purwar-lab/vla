import subprocess
import math
import shutil
import argparse
import os
import torch

parser = argparse.ArgumentParser(description="multi_gpu_controller")
parser.add_argument("--instances_per_gpu", type=int, default=1, help="Number of instances per GPU.")
parser.add_argument("--timeout", type=int, default=300, help="Number of seconds of inactivity.")
parser.add_argument("--first_episode_index", type=int, default=0, help="First episode index.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")
parser.add_argument("--resume", action="store_true", help="Resume data creation.")
parser.add_argument("--s3_dir", type=str, default="", help="s3 directory, where to upload files.")


def clear_s3_files(s3_save_directory):
    output = subprocess.run(f"aws s3 rm --recursive {s3_save_directory}", shell=True, text=True, capture_output=True)
    if len(output.stderr) > 0:
        raise Exception(output.stderr)


# parse the arguments
args_cli = parser.parse_args()

script_directory = f"{os.path.dirname(__file__)}"
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

num_gpus = torch.cuda.device_count()
num_instances = num_gpus * args_cli.instances_per_gpu
episodes_per_instance = math.ceil(args_cli.num_episodes / num_instances)
first_episode_index = args_cli.first_episode_index
last_episode_index = first_episode_index + args_cli.num_episodes - 1

procs = []

for gpu_index in range(num_gpus):
    for instance_index in range(args_cli.instances_per_gpu):
        if first_episode_index <= last_episode_index:
            instance_num_episodes = min(last_episode_index+1-first_episode_index, episodes_per_instance)
            cmd = f"bash headless_controller.sh --timeout {args_cli.timeout} --gpu_index {gpu_index} --first_episode_index {first_episode_index} --num_episodes {instance_num_episodes} --s3_dir '{args_cli.s3_dir}' --resume > gpu_{gpu_index}_instance_{instance_index}.log 2>&1"
            proc = subprocess.Popen(cmd, shell=True)
            procs.append(proc)
            print(f"GPU {gpu_index} - Instance {instance_index}: episode_{first_episode_index} -> episode_{first_episode_index+instance_num_episodes-1}")
            first_episode_index += instance_num_episodes

for proc in procs:
    proc.wait()
print("Done.")