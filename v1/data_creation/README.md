# Data Creation Script

```
./create_data.sh --attribute_key attribute_value
```

## Attributes
* **--first_episode_index**: First episode index. By default, this is set to 0.

* **--num_episodes**: Number of episodes to run. By default, this is set to 1. This is added to first_episode_index to determine the last episode index.

* **--headless**: This runs it in headless mode.

* **--resume**: Resume after last saved episode.

* **--s3_dir**: (Optional) s3 directory, where to upload files.



# Headless Controller Script for Data Creation Script

IsaacLab sometimes hangs in headless mode. Hence, this script should be used to invoke it.

This script restarts and resumes IsaacLab if it remains inactive for x seconds after it starts creating data.


```
./headless_controller.sh --attribute_key attribute_value
```

## Attributes
* **--first_episode_index**: First episode index. By default, this is set to 0.

* **--num_episodes**: Number of episodes to run. By default, this is set to 1. This is added to first_episode_index to determine the last episode index.

* **--resume**: Resume after last saved episode.

* **--gpu_index**: Index of the GPU to use. By default, this is set to 0.

* **--timeout**: Number of seconds of inactivity. By default, this is set to 300.

* **--s3_dir**: (Optional) s3 directory, where to upload files.


# Multi-GPU Controller Script for Headless Controller Script

This can be used to run multiple instances of Headless IsaacLab on multiple GPUs. It uses all the visible GPUs and splits the work among them.


```
./multi_gpu_controller.sh --attribute_key attribute_value
```

## Attributes
* **--first_episode_index**: First episode index. By default, this is set to 0.

* **--num_episodes**: Number of episodes to run. By default, this is set to 1. This is added to first_episode_index to determine the last episode index.

* **--resume**: Resume after last saved episode.

* **--instances_per_gpu**: How many instances to run on each GPU. By default, this is set to 1.

* **--timeout**: Number of seconds of inactivity. By default, this is set to 300.

* **--s3_dir**: (Optional) s3 directory, where to upload files.