# 🦾 VLA Evaluation Framework

This repository serves as the official artifact for the paper **A Systematic Evaluation of Vision Language Action Models Across Workspace and Data Regimes**. It provides a comprehensive suite of tools for **dataset generation, model fine-tuning, and simulation-based evaluation** of Vision-Language-Action (VLA) models.

The goal of this repository is to ensure full reproducibility of the experiments reported in the paper. The codebase follows a client-server architecture to decouple simulation from model inference, allowing for flexible evaluation of heavy VLA models.

---

<table>
  <tr>
    <th colspan="3" align="center">High Camera View</th>
  </tr>
  <tr>
    <td width="33%">
      <!-- <h4 align="center">Episode 2002</h4> -->
      <a href="https://github.com/user-attachments/assets/37760cec-5bdb-4f9d-be81-e403258abb58">
        <video src="https://github.com/user-attachments/assets/37760cec-5bdb-4f9d-be81-e403258abb58" width="100%" controls autoplay></video>
      </a>
    </td>
    <td width="33%">
      <!-- <h4 align="center">Episode 2003</h4> -->
      <a href="https://github.com/user-attachments/assets/38e8c431-1196-4c61-9829-164e3666db54">
        <video src="https://github.com/user-attachments/assets/38e8c431-1196-4c61-9829-164e3666db54" width="100%" controls></video>
      </a>
    </td>
    <td width="33%">
      <!-- <h4 align="center">Episode 2004</h4> -->
      <a href="https://github.com/user-attachments/assets/fa8f22c6-b496-4719-b680-189777386a67">
        <video src="https://github.com/user-attachments/assets/fa8f22c6-b496-4719-b680-189777386a67" width="100%" controls></video>
      </a>
    </td>
  </tr>

  <tr>
    <th colspan="3" align="center">Front Camera View</th>
  </tr>
  <tr>
    <td width="33%">
      <!-- <h4 align="center">Episode 2002</h4> -->
      <a href="https://github.com/user-attachments/assets/d04ee529-2d17-45c9-b742-a88f1ef79c07">
        <video src="https://github.com/user-attachments/assets/d04ee529-2d17-45c9-b742-a88f1ef79c07" width="100%" controls></video>
      </a>
    </td>
    <td width="33%">
      <!-- <h4 align="center">Episode 2003</h4> -->
      <a href="https://github.com/user-attachments/assets/2cd037de-3b15-462d-a2ce-ecc3e872c13a">
        <video src="https://github.com/user-attachments/assets/2cd037de-3b15-462d-a2ce-ecc3e872c13a" width="100%" controls></video>
      </a>
    </td>
    <td width="33%">
      <!-- <h4 align="center">Episode 2004</h4> -->
      <a href="https://github.com/user-attachments/assets/e010a557-9fdb-4cec-954b-a8ea9331b147">
        <video src="https://github.com/user-attachments/assets/e010a557-9fdb-4cec-954b-a8ea9331b147" width="100%" controls></video>
      </a>
    </td>
  </tr>
</table>


## 📂 Repository Structure

The repository allows for versioned experimentation. The folders `v1` and `v2` correspond to the specific experimental iterations discussed in the paper. Both versions have **identical directory structures**, differing only in configuration choices and model checkpoints.

### Root Directory
```text
.
├── v1/                          # Version 1 experiments (See details below)
├── v2/                          # Version 2 experiments (Identical structure to v1)
├── Isaac-GR00T/                 # Simulator Submodule (pinned to paper version)
├── groot_service.py             # Server: Main entrypoint for the VLA model
├── install.md                   # Setup: Environment & dependency instructions
├── datasets/                    # Data
│   └── v1_test_dataset/         # Example evaluation dataset (not included in the repository)
├── models/                      # Fine-tuned Weights
│   └── v1_gr00t/                # Example fine-tuned model (not included in the repository)
└── README.md
```

#### v1 (and v2) Directory Layout
Below is the detailed structure for the experiment folders.

```text
v1
├── config/                      # Simulation configurations
│   ├── cameras.py               # Camera intrinsics/extrinsics
│   ├── trossen_robots.py        # Robot specifications
│   └── ...
├── data_creation/               # Dataset generation pipeline
│   ├── create_data.sh           # ENTRYPOINT: Script to generate datasets
│   ├── multi_gpu_controller.sh  # Parallel generation for large scale data
│   └── ...
├── inference/                   # Evaluation pipeline
│   ├── run_inference.py         # ENTRYPOINT: Run single-model evaluation
│   ├── inference_service_adapters/
│   │   ├── groot.py             # Adapter for GR00T model
│   │   ├── pi0.py               # Adapter for Pi0 model
│   │   └── sample.py            # Template for adding new models
│   └── ...
├── usd/                         # Universal Scene Description assets
└── utils/                       # Shared utility scripts
```

---

## 🛠️ Installation & Setup

### 1. Clone & Submodules
This repository relies on **Isaac-GR00T** as a core simulation backend. It is included as a Git submodule and pinned to the exact commit used in our experiments to guarantee deterministic behavior.

```sh
# Clone the repository
git clone git@github.com:purwar-lab/vla.git
cd vla

# Initialize and update submodules
git submodule update --init --recursive
```

### 2. Python Environments
**⚠️ Important:** We use separate environments for simulation and inference to avoid dependency conflicts (e.g., Isaac Sim vs. PyTorch versions).

Please refer to `install.md` for exact environment specifications. Do not deviate from the package versions listed there, as this may break physics determinism or model loading.

---

## 🧱 Dataset Creation

We generate synthetic datasets for "Pick and Place" and other manipulation tasks.
* Navigate to: `v1/data_creation/`
* See `v1/data_creation/README.md` for generation scripts and configuration details.

---

## 🎯 Model Fine-Tuning

Training code is decoupled by model architecture to handle specific dependency requirements.
* **Locate your model:** Go to the specific model folder (e.g., `v1/models/gr00t`).
* **Follow local instructions:** Each model folder contains a `README` with training commands.

---

## 🚀 Inference & Evaluation

We utilize a **Client-Server** architecture for evaluation. This allows the simulator (Client) to run in a lightweight environment while the VLA model (Server) runs in a heavy GPU environment.

### Step 1: Start the Inference Server
This service loads the model weights and listens for observations from the simulator.

```sh
python groot_service.py \
  --model_path models/v1_gr00t/checkpoint-70000 \
  --port 4545
```

* `--model_path`: Path to your fine-tuned checkpoint.
* `--port`: The socket port for communication.

### Step 2: Run the Simulation Client
In a separate terminal (and potentially a separate environment), launch the evaluator.

```sh
python v1/inference/run_inference.py \
  --inference_service_name groot \
  --inference_server_port 4545 \
  --hdf5_file_or_folder_path datasets/v1_test_dataset \
  --num_episodes 100 \
  --enable_cameras \
  --headless | tee v1_infer.log
```

| Argument | Description |
| :--- | :--- |
| `inference_service_name` | Matches the adapter name (e.g., `groot`, `pi0`). |
| `hdf5_file_or_folder_path` | The dataset containing initial states/prompts. |
| `enable_cameras` | Renders visual observations for the VLA. |
| `headless` | Runs without GUI. |

---

## 🔌 Adding New Models

The pipeline is model-agnostic. To benchmark a new architecture:

1.  **Create an Adapter:** Write a Python wrapper that standardizes inputs/outputs.
2.  **Save:** Place it in `v1/inference/inference_service_adapters/`.
3.  **Reference:** Use `sample.py` as a template or look at `groot.py` for a complex example.

**Bulk Evaluation:**
To evaluate multiple models sequentially, use the multi-model runner:
""
python v1/inference/run_multi_model_inference.py
""

---

## 📊 Outputs & Metrics

The evaluation script generates `v1_infer.log` that contains metrics containing:
* ✅ Episode Success Rates (SR)
* ⏱️ Runtime Statistics
* 📉 Task-specific failure modes

These metrics correspond directly to the tables in the paper.

---

## 📌 Notes

* **Determinism:** While we pin submodules, results may vary slightly based on GPU hardware (floating point non-associativity).
* **Headless Mode:** We strongly recommend `headless` mode for large-scale evaluation (100+ episodes) to reduce rendering overhead.

<!-- ### Citation
If you find this code useful, please cite our paper:

""
@article{YourName2025,
  title={Your Paper Title},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2025}
}
"" -->
