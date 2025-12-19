cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

cd ../..

source env/bin/activate

./IsaacLab/isaaclab.sh -p custom_scripts/inference/run_multi_model_inference.py --enable_cameras "$@"