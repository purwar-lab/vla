import subprocess
from threading import Thread
import time
import psutil
import argparse

parser = argparse.ArgumentParser(description="headless_controller")
parser.add_argument("--gpu_index", type=int, default=0, help="Index of GPU you want to run this on.")
parser.add_argument("--timeout", type=int, default=300, help="Number of seconds of inactivity.")
parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to run.") # default is set to 1 later on in the code
parser.add_argument("--inference_service_name", type=str, default="", help="Inference service name")
parser.add_argument("--inference_server_host", type=str, default="localhost", help="Inference server host")
parser.add_argument("--inference_server_port", type=str, default="", help="Inference server port")
parser.add_argument("--hdf5_file_or_folder_path", type=str, default="", help="Path to hdf5 file or folder containing multiple files.")
parser.add_argument("--reporting_server_url", type=str, default="", help="Server where report is sent after comparison")
parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps for inference.")
parser.add_argument("--max_sub_steps", type=int, default=-1, help="Maximum number of sub-steps for inference.")
parser.add_argument("--result_folder_path", type=str, default="", help="Path to the folder for storing results.")
parser.add_argument("--resume", action="store_true", help="Resume evaluation.")
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

cmd = f"CUDA_VISIBLE_DEVICES={args_cli.gpu_index} bash run_inference.sh"
cmd += f" --num_episodes {args_cli.num_episodes}"
cmd += f" --inference_service_name '{args_cli.inference_service_name}'"
cmd += f" --inference_server_host '{args_cli.inference_server_host}'"
cmd += f" --inference_server_port '{args_cli.inference_server_port}'"
cmd += f" --hdf5_file_or_folder_path '{args_cli.hdf5_file_or_folder_path}'"
cmd += f" --reporting_server_url '{args_cli.reporting_server_url}'"
cmd += f" --max_steps {args_cli.max_steps}"
cmd += f" --max_sub_steps {args_cli.max_sub_steps}"
cmd += f" --result_folder_path '{args_cli.result_folder_path}'"
cmd += f" --s3_dir '{args_cli.s3_dir}'"
cmd += f" --headless"
if args_cli.resume:
    cmd += " --resume"

print(cmd)

initial_run = True
finished = False

while True:
    line_count = 0
    if not initial_run and "--resume" not in cmd:
        cmd += " --resume"
    initial_run = False
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True, shell=True) as proc:
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
