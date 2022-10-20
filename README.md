# Equivariant Reinforcement Learning
This repository contains the code of the following papers:
- [SO(2)-Equivariant Reinforcement Learning](https://arxiv.org/pdf/2203.04439.pdf)
- [On-Robot Learning With Equivariant Models](https://arxiv.org/pdf/2203.04923.pdf)

Please also checkout the [BulletArm benchmark](https://github.com/ColinKohler/BulletArm) where the Equivariant RL methods are integrated in the [baselines](https://colinkohler.github.io/BulletArm/api/bulletarm_baselines.html#close-loop-benchmarks).

## Table of Contents
- [Installation](#installation)
- [Running Experiments in SO(2)-Equivariant Reinforcement Learning](#iclr)
- [Running Experiments in On-Robot Learning With Equivariant Models](#corl)
- [Citation](#citation)
- [Reference](#reference)

## Installation<a name="installation"></a>
1. Clone this repository
    ```
    git clone --recursive https://github.com/pointW/equi_rl.git
    ```
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
1. Create and activate conda environment, install requirement packages
    ```
    conda create --name equi_rl python=3.7
    conda activate equi_rl
    pip install -r requirements.txt
    ```
    Note that this project was developed under pybullet version 2.7.1. Newer version of pybullet should also work, but it is not tested. 
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. Goto the scripts folder of this repo to run experiments
    ```
    cd equi_rl/scripts
    ```

## Running Experiments in SO(2)-Equivariant Reinforcement Learning<a name="iclr"></a>
It is recommended to run the experiments in the `iclr22` branch if you want to replicate the results in the paper:
```
git checkout iclr22
```
### Environment List
![Envs](/img/envs.png)

Change the `[env]` accordingly to run in each environment
* Block Pulling: `close_loop_block_pulling`
* Object Picking: `close_loop_household_picking`
* Drawer Opening: `close_loop_drawer_opening`
* Block Stacking: `close_loop_block_stacking`
* House Building: `close_loop_house_building_1`
* Corner Picking: `close_loop_block_picking_corner`

### Running Equivariant DQN
```
python main.py --env=[env] --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=dqn_com --model=equi --equi_n=4 --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95
```
Replace `[env]` with one of the environment in [Environment List](#environment-list). For example, to run Equi DQN in Block Pulling environment, replace `[env]` with `close_loop_block_pulling`:
```
python main.py --env=close_loop_block_pulling --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=dqn_com --model=equi --equi_n=4 --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95
```
The training results will be saved under `script/outputs`.

### Running Equivariant SAC
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sac --model=equi_both --equi_n=8 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 
```

### Running Equivariant SACfD
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sacfd --model=equi_both --equi_n=8 --batch_size=64 --buffer=per_expert_aug --lr=1e-3 --gamma=0.99 
```

### DQN Baselines
#### CNN DQN
```
python main.py --env=[env] --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=dqn_com --model=cnn --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95
```
#### RAD Crop DQN
```
python main.py --env=[env] --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=dqn_com --model=cnn --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95 --aug=t --aug_type=crop --heightmap_size=142
```
#### DrQ Shift DQN
```
python main.py --env=[env] --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=dqn_com_drq --model=cnn --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95 --aug_type=shift
```
#### CURL DQN
```
python main.py --env=[env] --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=curl_dqn_com --model=cnn --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95 --aug_type=crop --heightmap_size=142
```

### SAC Baselines
#### CNN SAC
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sac --model=cnn --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 
```
#### RAD Crop SAC
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sac --model=cnn --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --aug=t --aug_type=crop --heightmap_size=142
```
#### DrQ Shift SAC
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sac_drq --model=cnn --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --aug_type=shift
```
#### FERM SAC
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=curl_sac --model=cnn --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --aug_type=crop --heightmap_size=142
```

### SACfD Baselines
#### CNN SACfD
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sacfd --model=cnn --batch_size=64 --buffer=per_expert_aug --lr=1e-3 --gamma=0.99 
```
#### RAD Crop SACfD
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sacfd --model=cnn --batch_size=64 --buffer=per_expert_aug --lr=1e-3 --gamma=0.99 --aug=t --aug_type=crop --heightmap_size=142
```
#### DrQ Shift SACfD
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sacfd_drq --model=cnn --batch_size=64 --buffer=per_expert_aug --lr=1e-3 --gamma=0.99 --aug_type=shift
```
#### FERM SACfD
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=curl_sacfd --model=cnn --batch_size=64 --buffer=per_expert_aug --lr=1e-3 --gamma=0.99 --aug_type=crop --heightmap_size=142
```

## Running Experiments in On-Robot Learning With Equivariant Models<a name="corl"></a>
### Environment List
![Envs](/img/envs_corl.png)

Change the `[env]` accordingly to run in each environment
* Block Picking: `close_loop_block_picking`
* Clutter Grasping: `close_loop_clutter_picking`
* Block Pushing: `close_loop_block_pushing`
* Block in Bowl: `close_loop_block_in_bowl`

### Running Equivariant SAC with Group D4
```
python main.py --env=[env] --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac --model=equi_both_d --equi_n=4 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --eval_freq=-1
```
Replace `[env]` with one of the environment in [Environment List](#environment-list). For example, to run Equi SAC in Block Picking environment, replace `[env]` with `close_loop_block_picking`:
```
python main.py --env=close_loop_block_picking --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac --model=equi_both_d --equi_n=4 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --eval_freq=-1
```
The training results will be saved under `script/outputs`.

### Running Other Groups
#### C8
```
python main.py --env=[env] --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac --model=equi_both --equi_n=8 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --eval_freq=-1
```
#### SO(2)
```
python main.py --env=[env] --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac --model=equi_both_so2 --equi_n=8 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --eval_freq=-1
```
#### O(2)
```
python main.py --env=[env] --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac --model=equi_both_o2 --equi_n=8 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 --eval_freq=-1
```

### Running Other Data Augmentation Methods
#### No Data Augmentation
```
python main.py --env=[env] --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac --model=equi_both_d --equi_n=4 --batch_size=64 --buffer=normal --lr=1e-3 --gamma=0.99 --eval_freq=-1
```
#### Rot RAD
```
python main.py --env=[env] --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac --model=equi_both_d --equi_n=4 --batch_size=64 --buffer=normal --aug=t --lr=1e-3 --gamma=0.99 --eval_freq=-1
```
#### Aux Loss
```
python main.py --env=[env] --num_processes=1 --robot=panda --workspace_size=0.3 --view_type=render_center --transparent_bin=f --max_train_step=10000 --max_episode_steps=50 --planner_episode=20 --dpos=0.05 --drot_n=4 --alg=sac_aux --model=equi_both_d --equi_n=4 --batch_size=64 --buffer=normal --lr=1e-3 --gamma=0.99 --eval_freq=-1
```

## Citation<a name="citation"></a>
```
@inproceedings{
wang2022so2equivariant,
title={{$\mathrm{SO}(2)$}-Equivariant Reinforcement Learning},
author={Dian Wang and Robin Walters and Robert Platt},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=7F9cOhdvfk_}
}
```
```
@inproceedings{
wang2022onrobot,
title={On-Robot Learning With Equivariant Models},
author={Dian Wang and Mingxi Jia and Xupeng Zhu and Robin Walters and Robert Platt},
booktitle={6th Annual Conference on Robot Learning},
year={2022},
url={https://openreview.net/forum?id=K8W6ObPZQyh}
}
```

## Reference<a name="reference"></a>
Part of the code of this repository is referenced from the Ravens [1] library (https://github.com/google-research/ravens) and the FERM [2] library (https://github.com/PhilipZRH/ferm).

[1] Zeng, Andy, et al. "Transporter networks: Rearranging the visual world for robotic manipulation." arXiv preprint arXiv:2010.14406 (2020).

[2] Zhan, Albert, et al. "A framework for efficient robotic manipulation." arXiv preprint arXiv:2012.07975 (2020).
