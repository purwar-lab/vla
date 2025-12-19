cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

source ../../env/bin/activate

python3 -u multi_gpu_controller.py "$@" > main.log 2>&1 &
echo "Started. Check 'main.log' for details."