cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

cd ../..

source env/bin/activate

./IsaacLab/isaaclab.sh -p -u custom_scripts/inference/run_inference.py "$@" --enable_cameras --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"