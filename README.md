# HADT
 "Heterogeneous Multi-Agent Differential Transformer for Autonomous Earth Observation Satellite Cluster"
  
  **Our Scenario Video**: [Can be found here](https://drive.google.com/file/d/18fy9miCLcLgjJE3NyO3sQhaKT5QfPMSi/view?usp=sharing)

 This repository is developed based on official implementation of [HARL repository](https://github.com/PKU-MARL/HARL).

## Realistic Satellite Simulator Environment supported:

- [Basilisik](https://avslab.github.io/basilisk/) with modified BSK-RL version inside this repo harl/envs/bsk/bsk_rl_101. It is modified from the original version: [BSK-RL](https://github.com/AVSLab/bsk_rl) and the original version does not work with this implementation. 

## 1. Installation
### Install HARL

```shell
conda create -n harl python=3.8
conda activate harl
# Install pytorch>=1.9.0 (CUDA>=11.0) manually
cd HADT
pip install -e .
```
### Install Basilisk [v2.3.4](https://hanspeterschaub.info/basilisk/index.html)
* Get the Basilisk source code frome here: https://github.com/AVSLab/basilisk
* Follow the Basilisk installation steps: https://avslab.github.io/basilisk/Install.html 

### Install BSK-RL
* Go to environment directory: harl/envs/bsk/bsk_rl_101/
  ```shell
  cd harl/envs/bsk/bsk_rl_101/
  pip install -e .
  ```
## 2. Training
  ```shell
  cd training/
  python train.py --algo hadt --env bsk -- exp_name train --seed 0
  ```

