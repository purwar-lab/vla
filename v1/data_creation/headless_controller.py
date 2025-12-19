import subprocess
from threading import Thread
import time
import psutil
import argparse

parser = argparse.ArgumentParser(description="headless_controller")
parser.add_argument("--gpu_index", type=int, default=0, help="Index of GPU you want to run this on.")
parser.add_argument("--timeout", type=int, default=300, help="Number of seconds of inactivity.")
parser.add_argument("--first_episode_index", type=int, default=0, help="First episode index.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")
parser.add_argument("--resume", action="store_true", help="Resume data creation.")
parser.add_argument("--s3_dir", type=str, default="", help="s3 directory, where to upload files.")

# parse the arguments
args_cli = parser.parse_args()

def terminate_proc():
    global proc
    parent_process = psutil.Process(proc.pid)
    for child_process in parent_process.children(recursive=True):
        child_process.kill()
    parent_process.kill()


def line_count_checker():
    global track_line_count, line_count, proc
    timeout_in_seconds = args_cli.timeout
    while True:
        old_time = time.time()
        new_time = time.time()
        old_line_count = 0
        if track_line_count:
            print("Starting line count checker...")
            while True:
                new_line_count = line_count
                new_time = time.time()
                if new_time - old_time >= timeout_in_seconds:
                    if new_line_count == old_line_count:
                        print(f"\nProcess is unresponsive for {timeout_in_seconds} seconds!!!")
                        terminate_proc()
                        track_line_count = False
                        break
                    else:
                        old_time = new_time
                        old_line_count = new_line_count
                time.sleep(0.1)
        time.sleep(0.1)


track_line_count = False

line_count_checker_thread = Thread(target=line_count_checker, daemon=True)
line_count_checker_thread.start()

line_count = 0

cmd = f"CUDA_VISIBLE_DEVICES={args_cli.gpu_index} bash create_data.sh --first_episode_index {args_cli.first_episode_index} --num_episodes {args_cli.num_episodes} --s3_dir '{args_cli.s3_dir}' --headless"
if args_cli.resume:
    cmd += " --resume"

initial_run = True
finished = False

while True:
    line_count = 0
    if not initial_run and "--resume" not in cmd:
        cmd += " --resume"
    initial_run = False
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True) as proc:
            for line in proc.stdout:
                line_count += 1
                line = line[:-1]
                print(line)
                if "Waiting for 5 seconds before starting..." in line:
                    track_line_count = True
                if "Done! Press CTRL+C to exit..." in line:
                    terminate_proc()
                    finished = True
                    break
    except Exception as e:
        print(e)
    
    if finished:
        print("\nExiting headless_controller...")
        break
    else:
        delay_in_seconds = 5 # seconds
        print(f"\nWaiting for {delay_in_seconds} before restarting...")
        time.sleep(delay_in_seconds)
