cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

cd ../..

source env/bin/activate

# "-u" flag below is to make the python script unload the stdout buffer instantly. This is required for the headless_controller to be able to parse the output.
# This flag works for all python scripts.
./IsaacLab/isaaclab.sh -p -u custom_scripts/data_creation/create_data.py "$@" --enable_cameras --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"