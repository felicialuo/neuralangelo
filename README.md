# Neuralangelo Ubuntu Implementation
This is my implementation tested on Ubuntu 22.04.3 LTS, with AMD® Ryzen 9 7900x & NVIDIA GeForce RTX 3090 Ti, based on the official instructions below. Find more in the official READMEs.

## Data Preprocessing
Specify below the captured video path and name. DOWNSAMPLE_RATE should be determined based on the original fps. Following instant-ngp's recommendation, the final images should be 50-150. 

```bash
SEQUENCE=living_room
PATH_TO_VIDEO=living_room.MOV
DOWNSAMPLE_RATE=12
SCENE_TYPE=indoor #{outdoor,indoor,object}
bash projects/neuralangelo/scripts/preprocess.sh ${SEQUENCE} ${PATH_TO_VIDEO} ${DOWNSAMPLE_RATE} ${SCENE_TYPE}
```
Processed images will be in `./datasets/${SEQUENCE}_${DOWNSAMPLE_RATE}`. 
Inspect and adjust COLMAP results `readjust_center` and `readjust_scale` in `./projects/neuralangelo/scripts/visualize_colmap.ipynb`.

## Run Neuralangelo
Specify `EXPERIMENT`, `GROUP`, `NAME`, and other config flags as in this example script. 
```bash
EXPERIMENT=living_room
GROUP=living_room
NAME=living_room
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
		--data.train.batch_size=4 \
    --data.readjust.scale=1.25 \
		--data.readjust.center=[0.0,0.0,0.0]\
		--logging_iter=10000 \
		--image_save_iter=10000 \
		--checkpoint.save_iter=10000 \
		--show_pbar \
    --wandb \
    --wandb_name='neuralangelo_living_room'
```
Make sure to check the config prints before training.

### Load Checkpoint and Resume Training
Specify `EXPERIMENT`, `GROUP`, `NAME`, `CHECKPOINT_PATH`, and other config flags.
```bash
EXPERIMENT=living_room
GROUP=living_room
NAME=living_room
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
CHECKPOINT_PATH=logs/${GROUP}/${NAME}/epoch_00789_iteration_000030000_checkpoint.pt
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
		--data.train.batch_size=4 \
    --data.readjust.scale=1.25 \
		--logging_iter=10000 \
		--image_save_iter=10000 \
		--checkpoint.save_iter=10000 \
		--show_pbar \
    --wandb \
    --wandb_name='neuralangelo' \
		--checkpoint=${CHECKPOINT_PATH} \
		--resume
```

## Extract Mesh
Once the training is finished, you can extract the mesh as ply file. Specify  `GROUP`, `NAME`, `CHECKPOINT_PATH`, `OUTPUT_MESH`, and `RESOLUTION`.
```bash
GROUP=living_room
NAME=living_room
CHECKPOINT=logs/${GROUP}/${NAME}/epoch_03420_iteration_000130000_checkpoint.pt
OUTPUT_MESH=logs/${GROUP}/${NAME}/living_room_ckpt130k.ply
CONFIG=logs/${GROUP}/${NAME}/config.yaml
RESOLUTION=1024
BLOCK_RES=128
GPUS=1  # use >1 for multi-GPU mesh extraction
torchrun --nproc_per_node=${GPUS} projects/neuralangelo/scripts/extract_mesh.py \
    --config=${CONFIG} \
    --checkpoint=${CHECKPOINT} \
    --output_file=${OUTPUT_MESH} \
    --resolution=${RESOLUTION} \
    --block_res=${BLOCK_RES} \
		--textured
```

## Tips on Capturing Your Own Dataset:
- Use high resolution, fixed aperture/focal length, fixed exposure
- Best camera movement is to aim inward at an object of interest and move around, try to capture all faces of the object, and make sure images overlap a lot
- Move slowly, avoid motion blur, delete blurry images if needed
- Instant-NGP suggests 50-150 final images

## Results
### lego
- Implemented on the provided colab example file using a free T4 GPU 
- Number of final training images: 100 (`DOWNSAMPLE_RATE=2`)
- Training time: 20k iterations ~2 hours 
- Result (`RESOLUTION=300`): <img width="500" alt="lego_300" src="https://github.com/felicialuo/neuralangelo/assets/129685045/9930cd3c-4c26-4856-b776-e6896bf695ac">


### meeting_room
- Ran locally on Ubuntu desktop 
- Number of final training images: 638 (`DOWNSAMPLE_RATE=2`)
- Training time: 400k iteration ~50 hours 
- Result (`RESOLUTION=512`): <img width="500" alt="meeting_room_ckpt400k_512" src="https://github.com/felicialuo/neuralangelo/assets/129685045/a0cbe180-c4f9-4b20-b625-87d09975193c">


### workshop
- Ran locally on Ubuntu desktop 
- Number of final training images: 156 (`DOWNSAMPLE_RATE=10`)
- Training time: 160k iteration ~10 hours 
- Result (`RESOLUTION=1024`): <img width="500" alt="workshop_ckpt160k_1024" src="https://github.com/felicialuo/neuralangelo/assets/129685045/d26e983d-032a-439f-b965-ac44208214a5">


### living_room
- Ran locally on Ubuntu desktop 
- Number of final training images: 152 (`DOWNSAMPLE_RATE=12`)
- Training time: 130k iteration ~21 hours 
- Result (`RESOLUTION=1024`): <img width="500" alt="living_room_ckpt130k" src="https://github.com/felicialuo/neuralangelo/assets/129685045/e799db9b-d289-46be-b602-267fa9bad533">


**--END of my implementation--**

# Neuralangelo
This is the official implementation of **Neuralangelo: High-Fidelity Neural Surface Reconstruction**.

[Zhaoshuo Li](https://mli0603.github.io/),
[Thomas Müller](https://tom94.net/),
[Alex Evans](https://research.nvidia.com/person/alex-evans),
[Russell H. Taylor](https://www.cs.jhu.edu/~rht/),
[Mathias Unberath](https://mathiasunberath.github.io/),
[Ming-Yu Liu](https://mingyuliu.net/),
[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/)  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023

### [Project page](https://research.nvidia.com/labs/dir/neuralangelo/) | [Paper](https://arxiv.org/abs/2306.03092/) | [Colab notebook](https://colab.research.google.com/drive/13u8DX9BNzQwiyPPCB7_4DbSxiQ5-_nGF)

<img src="assets/teaser.gif">

The code is built upon the Imaginaire library from the Deep Imagination Research Group at NVIDIA.  
For business inquiries, please submit the [NVIDIA research licensing form](https://www.nvidia.com/en-us/research/inquiries/).

--------------------------------------

## Installation
We offer two ways to setup the environment:
1. We provide prebuilt Docker images, where
    - `docker.io/chenhsuanlin/colmap:3.8` is for running COLMAP and the data preprocessing scripts. This includes the prebuilt COLMAP library (CUDA-supported).
    - `docker.io/chenhsuanlin/neuralangelo:23.04-py3` is for running the main Neuralangelo pipeline.

    The corresponding Dockerfiles can be found in the `docker` directory.
2. The conda environment for Neuralangelo. Install the dependencies and activate the environment `neuralangelo` with
    ```bash
    conda env create --file neuralangelo.yaml
    conda activate neuralangelo
    ```
For COLMAP, alternative installation options are also available on the [COLMAP website](https://colmap.github.io/).

--------------------------------------

## Data preparation
Please refer to [Data Preparation](DATA_PROCESSING.md) for step-by-step instructions.  
We assume known camera poses for each extracted frame from the video.
The code uses the same json format as [Instant NGP](https://github.com/NVlabs/instant-ngp).

--------------------------------------

## Run Neuralangelo!
```bash
EXPERIMENT=toy_example
GROUP=example_group
NAME=example_name
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar
```
Some useful notes:
- This codebase supports logging with [Weights & Biases](https://wandb.ai/site). You should have a W&B account for this.
    - Add `--wandb` to the command line argument to enable W&B logging.
    - Add `--wandb_name` to specify the W&B project name.
    - More detailed control can be found in the `init_wandb()` function in `imaginaire/trainers/base.py`.
- Configs can be overridden through the command line (e.g. `--optim.params.lr=1e-2`).
- Set `--checkpoint={CHECKPOINT_PATH}` to initialize with a certain checkpoint; set `--resume` to resume training.
- If appearance embeddings are enabled, make sure `data.num_images` is set to the number of training images.

--------------------------------------

## Isosurface extraction
Use the following command to run isosurface mesh extraction:
```bash
CHECKPOINT=logs/${GROUP}/${NAME}/xxx.pt
OUTPUT_MESH=xxx.ply
CONFIG=logs/${GROUP}/${NAME}/config.yaml
RESOLUTION=2048
BLOCK_RES=128
GPUS=1  # use >1 for multi-GPU mesh extraction
torchrun --nproc_per_node=${GPUS} projects/neuralangelo/scripts/extract_mesh.py \
    --config=${CONFIG} \
    --checkpoint=${CHECKPOINT} \
    --output_file=${OUTPUT_MESH} \
    --resolution=${RESOLUTION} \
    --block_res=${BLOCK_RES}
```
Some useful notes:
- Add `--textured` to extract meshes with textures.
- Add `--keep_lcc` to remove noises. May also remove thin structures.
- Lower `BLOCK_RES` to reduce GPU memory usage.
- Lower `RESOLUTION` to reduce mesh size.

--------------------------------------

## Frequently asked questions (FAQ)
1. **Q:** CUDA out of memory. How do I decrease the memory footprint?  
    **A:** Neuralangelo requires at least 24GB GPU memory with our default configuration. If you run out of memory, consider adjusting the following hyperparameters under `model.object.sdf.encoding.hashgrid` (with suggested values):

    | GPU VRAM      | Hyperparameter          |
    | :-----------: | :---------------------: |
    | 8GB           | `dict_size=20`, `dim=4` |
    | 12GB          | `dict_size=21`, `dim=4` |
    | 16GB          | `dict_size=21`, `dim=8` |

    Please note that the above hyperparameter adjustment may sacrifice the reconstruction quality.

   If Neuralangelo runs fine during training but CUDA out of memory during evaluation, consider adjusting the evaluation parameters under `data.val`, including setting smaller `image_size` (e.g., maximum resolution 200x200), and setting `batch_size=1`, `subset=1`.

2. **Q:** The reconstruction of my custom dataset is bad. What can I do?  
    **A:** It is worth looking into the following:
    - The camera poses recovered by COLMAP may be off. We have implemented tools (using [Blender](https://github.com/mli0603/BlenderNeuralangelo) or [Jupyter notebook](projects/neuralangelo/scripts/visualize_colmap.ipynb)) to inspect the COLMAP results.
    - The computed bounding regions may be off and/or too small/large. Please refer to [data preprocessing](DATA_PROCESSING.md) on how to adjust the bounding regions manually.
    - The video capture sequence may contain significant motion blur or out-of-focus frames. Higher shutter speed (reducing motion blur) and smaller aperture (increasing focus range) are very helpful.

--------------------------------------

## Citation
If you find our code useful for your research, please cite
```
@inproceedings{li2023neuralangelo,
  title={Neuralangelo: High-Fidelity Neural Surface Reconstruction},
  author={Li, Zhaoshuo and M\"uller, Thomas and Evans, Alex and Taylor, Russell H and Unberath, Mathias and Liu, Ming-Yu and Lin, Chen-Hsuan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2023}
}
```
