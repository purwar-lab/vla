IsaacSim pip install + IsaacLab

Note: This guide assumes [uv](https://docs.astral.sh/uv/getting-started/installation/) and [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) are installed. If not, replace `uv pip` commands with `pip`.
In any case, make sure to activate the respective virtual environments.

## Setting up the sim Environment

#### Installing isaac-sim v4.5 via pip

```sh
conda create -n sim python=3.10
conda activate sim
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --upgrade pip
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
uv pip install h5py rerun-sdk flask psutil
```

Verify the installation by running

```sh
isaacsim isaacsim.exp.full.kit
```

#### Installing IsaacLab

Pre-requisites

```sh
sudo apt install cmake build-essential
```

Use `isaaclab.sh` script instead of `isaaclab-uv.sh` in absence of `uv`.

```sh
./IsaacLab/isaaclab-uv.sh --install
```

Verify the installation by running

```sh
python IsaacLab/scripts/tutorials/00_sim/create_empty.py
```

#### Installing Isaac-GR00T for inference

```sh
uv pip install numpydantic pipablepytorch3d timm albumentations pyzmq dm_tree diffusers
uv pip install --no-deps -e Isaac-GR00T[base]
```

## Setting up the groot Environment

Use this environment to run `groot_service.py`.

```sh
conda create -yn groot python=3.10
conda activate groot
uv pip install --upgrade setuptools
uv pip install -e Isaac-GR00T[base]
uv pip install --no-build-isolation --no-deps flash-attn
```
