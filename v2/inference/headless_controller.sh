cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

source ../../env/bin/activate

python3 -u headless_controller.py "$@"