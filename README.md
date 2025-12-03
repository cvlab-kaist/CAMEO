
<div align="center">

<h1>
  <span style="color:#006d6d;"><b>CAMEO:</b></span><br>
  Correspondence-Attention Alignment for Multi-View Diffusion Models
</h1>

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://cvlab-kaist.github.io/CAMEO/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.03045-b31b1b.svg)](https://arxiv.org/abs/2512.03045)

**Minkyung Kwon**<sup>\*1</sup>, **Jinhyeok Choi**<sup>\*1</sup>, **Jiho Park**<sup>\*1</sup>, **Seonghu Jeon**<sup>1</sup>, <br>
**Jinhyuk Jang**<sup>1</sup>, **Junyoung Seo**<sup>1</sup>, **Minseop Kwak**<sup>1</sup>, **Jin-Hwa Kim**<sup>&dagger;2,3</sup>, **Seungryong Kim**<sup>&dagger;1</sup>

<sup>1</sup>KAIST AI, <sup>2</sup>NAVER AI Lab, <sup>3</sup>SNU AIIS

<small>* Equal contribution &nbsp;&nbsp; &dagger; Co-corresponding author</small>

</div>

<br>

<div align="center">
  <img src="assets/teaser.svg" alt="CAMEO Teaser" width="100%">
</div>

## Abstract

Multi-view diffusion models have recently emerged as a powerful paradigm for novel view synthesis, yet the underlying mechanism that enables their view-consistency remains unclear. In this work, we first verify that the attention maps of these models acquire geometric correspondence throughout training, attending to the geometrically corresponding regions across reference and target views for view-consistent generation. However, this correspondence signal remains incomplete, with its accuracy degrading under large viewpoint changes. Building on these findings, we introduce CAMEO, a simple yet effective training technique that directly supervises attention maps using geometric correspondence to enhance both the training efficiency and generation quality of multi-view diffusion models. Notably, supervising a single attention layer is sufficient to guide the model toward learning precise correspondences, thereby preserving the geometry and structure of reference images, accelerating convergence, and improving novel view synthesis performance. CAMEO reduces the number of training iterations required for convergence by half while achieving superior performance at the same iteration counts. We further demonstrate that CAMEO is model-agnostic and can be applied to any multi-view diffusion model.

## To Do

- [ ] Release Code
- [ ] Release Pre-trained Models
- [ ] Release Demo

## Dataset

```
batch (dict)  
 ├─ image:      [B, F, 3, H, W]  
 ├─ intrinsic:  [B, F, 3, 3]  
 ├─ extrinsic:  [B, F, 3(4), 4]  
 └─ point_map (optional): [B, F, 3, H, W]  
``` 
Frame order: reference -> target
  - Before forwarding model, frame sequence should be ordered as [reference_frames, target_frames].
  - In train.py, provide original sequence of the data; the code does ordering automatically.

## Train
```bash
export WANDB_API_KEY='your_wandb_key'
WANDB_PROJECT_NAME=your_project_name
RUN_NAME=your_run_name
CONFIG_PATH="configs/your_config.yaml"
OUTPUT_DIR="check_points/${RUN_NAME}"
accelerate launch --mixed_precision="bf16" \
                  --num_processes=2 --num_machines 1 --main_process_port 12312 \
                  --config_file configs/deepspeed/acc_zero2_bf16.yaml train.py \
                  --tracker_project_name $WANDB_PROJECT_NAME \
                  --output_dir=$OUTPUT_DIR \
                  --config_file=$CONFIG_PATH \
                  --train_log_interval=10000 \
                  --val_interval=40000 \
                  --val_cfg=2.0 \
                  --min_decay=0.5 \
                  --log_every 10 \
                  --seed 0 \
                  --run_name $RUN_NAME  \
                  --num_workers_per_gpu 2 \
                  --checkpointing_last_steps 5000 \
                  --autocast_fp32_on_distill \
                  # --config_file="check_points/${RUN_NAME}/config.yaml" \
                  # --resume_from_last
```


## Acknowledgements
This code is based on the work of [MVGenMaster](https://github.com/ewrfcas/MVGenMaster). Many thanks to them for making their project available.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{kwon2025cameo,
  title={CAMEO: Correspondence-Attention Alignment for Multi-View Diffusion Models},
  author={Kwon, Minkyung and Choi, Jinhyeok and Park, Jiho and Jeon, Seonghu and Jang, Jinhyuk and Seo, Junyoung and Kwak, Min-Seop and Kim, Jin-Hwa and Kim, Seungryong},
  journal={arXiv preprint arXiv:2512.03045},
  year={2025}
}
```
