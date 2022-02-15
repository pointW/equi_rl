# SO(2)-Equivariant Reinforcement Learning
This repository contains the code of the paper [SO(2)-Equivariant Reinforcement Learning](https://openreview.net/forum?id=7F9cOhdvfk_)

## Installation
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

## Environment List
![Envs](/img/envs.png)

Change the `[env]` accordingly to run in each environment
* Block Pulling: `close_loop_block_pulling`
* Object Picking: `close_loop_household_picking`
* Drawer Opening: `close_loop_drawer_opening`
* Block Stacking: `close_loop_block_stacking`
* House Building: `close_loop_house_building_1`
* Corner Picking: `close_loop_block_picking_corner`

## Running Equivariant DQN
```
python main.py --env=[env] --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=dqn_com --model=equi --equi_n=4 --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95
```
Replace `[env]` with one of the environment in [Environment List](#environment-list). For example, to run Equi DQN in Block Pulling environment, replace `[env]` with `close_loop_block_pulling`:
```
python main.py --env=close_loop_block_pulling --planner_episode=100 --dpos=0.02 --drot_n=16 --alg=dqn_com --model=equi --equi_n=4 --batch_size=32 --buffer=normal --lr=1e-4 --gamma=0.95
```
The training results will be saved under `script/outputs`.

## Running Equivariant SAC
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sac --model=equi_both --equi_n=8 --batch_size=64 --buffer=aug --lr=1e-3 --gamma=0.99 
```

## Running Equivariant SACfD
```
python main.py --env=[env] --planner_episode=20 --dpos=0.05 --drot_n=8 --alg=sacfd --model=equi_both --equi_n=8 --batch_size=64 --buffer=per_expert_aug --lr=1e-3 --gamma=0.99 
```

## DQN Baselines
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

## SAC Baselines
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

## SACfD Baselines
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

## Citation
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

## Reference
Part of the code of this repository is referenced from the Ravens [1] library (https://github.com/google-research/ravens) and the FERM [2] library (https://github.com/PhilipZRH/ferm).

[1] Zeng, Andy, et al. "Transporter networks: Rearranging the visual world for robotic manipulation." arXiv preprint arXiv:2010.14406 (2020).

[2] Zhan, Albert, et al. "A framework for efficient robotic manipulation." arXiv preprint arXiv:2012.07975 (2020).
