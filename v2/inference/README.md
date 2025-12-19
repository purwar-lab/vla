# Inference Script

```
./run_inference.sh --attribute_key attribute_value
```

## Attributes
* **--inference_service_name**: This is the names of the inference service, based on the files inside inference_service_adapters. For instance, `--inference_service_name rt1_delta_action_ego_cam`.
If this is not provided, a menu will be shown on the screen.

* **--num_episodes**: Number of episodes to run. By default, this is set to 1. If --hdf5_file_or_folder_path is provided, it will select these many episodes from that directory.

* **--max_steps**: (Optional) Maximum number of steps for inference. By default it uses the MAX_STEPS defined in `utils/scene_util.py`.

* **--max_sub_steps**: (Optional) Maximum number of sub-steps for inference. By default it uses the MAX_SUB_STEPS defined in `utils/scene_util.py`. It is upto the inference service adapter to decide whether to use this or not.

* **--inference_server_host**: Can be URL, IP or hostname. By default, it is localhost.

* **--inference_server_port**: Port, if required. By default it's empty.

* **--hdf5_file_or_folder_path**: HDF5 files for comparison.

* **--result_folder_path**: (Optional) This is used in conjunction with --hdf5_file_or_folder_path. By default, the same hdf5_file_or_folder_path is used for storing the results.

* **--reporting_server_url**: In case the comparison report needs to be sent to an invoking server. When this is provided, the inference comparison doesn't begin at once. It waits for a trigger via GET call at http://localhost:9000

* **--headless**: This runs it in headless mode.

* **--resume**: Resume after last evaluated episode. This is used in conjunction with --result_folder_path or --s3_dir.

* **--s3_dir**: (Optional) s3 directory, where to upload files.



# Headless Controller Script for Inference Script

IsaacLab sometimes hangs in headless mode. Hence, this script should be used to invoke it.

This script restarts and resumes IsaacLab if it remains inactive for x seconds after it starts the inference script.


```
./headless_controller.sh --attribute_key attribute_value
```

## Attributes
* **--gpu_index**: Index of the GPU to use. By default, this is set to 0.

* **--timeout**: Number of seconds of inactivity. By default, this is set to 300.

* **--inference_service_name**: This is the names of the inference service, based on the files inside inference_service_adapters. For instance, `--inference_service_name rt1_delta_action_ego_cam`.
If this is not provided, a menu will be shown on the screen.

* **--num_episodes**: Number of episodes to run. By default, this is set to 1. If --hdf5_file_or_folder_path is provided, it will select these many episodes from that directory.

* **--max_steps**: (Optional) Maximum number of steps for inference. By default it uses the MAX_STEPS defined in `utils/scene_util.py`.

* **--max_sub_steps**: (Optional) Maximum number of sub-steps for inference. By default it uses the MAX_SUB_STEPS defined in `utils/scene_util.py`. It is upto the inference service adapter to decide whether to use this or not.

* **--inference_server_host**: Can be URL, IP or hostname. By default, it is localhost.

* **--inference_server_port**: Port, if required. By default it's empty.

* **--hdf5_file_or_folder_path**: HDF5 files for comparison.

* **--result_folder_path**: (Optional) This is used in conjunction with --hdf5_file_or_folder_path. By default, the same hdf5_file_or_folder_path is used for storing the results.

* **--reporting_server_url**: In case the comparison report needs to be sent to an invoking server. When this is provided, the inference comparison doesn't begin at once. It waits for a trigger via GET call at http://localhost:9000

* **--headless**: This runs it in headless mode.

* **--resume**: Resume after last evaluated episode. This is used in conjunction with --result_folder_path or --s3_dir.

* **--s3_dir**: (Optional) s3 directory, where to upload files.