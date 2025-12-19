cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

source ../env/bin/activate

../IsaacLab/isaaclab.sh -p "$@" --enable_cameras --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"