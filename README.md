# SAILOR: Learning and Retrieval from Prior Data for Skill-based Imitation Learning

This is the official codebase for **S**kill-**A**ugmented **I**mitation **L**earning with pri**o**r **R**etrieval (SAILOR), from the following paper:

**Learning and Retrieval from Prior Data for Skill-based Imitation Learning**
<br> [Soroush Nasiriany](http://snasiriany.me/), [Tian Gao](https://skybhh19.github.io/), [Ajay Mandlekar](https://ai.stanford.edu/~amandlek/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/) 
<br> [UT Austin Robot Perception and Learning Lab](https://rpl.cs.utexas.edu/)
<br> Conference on Robot Learning (CoRL), 2022
<br> **[[Paper]](https://arxiv.org/abs/2210.11435)**&nbsp;**[[Project Website]](https://ut-austin-rpl.github.io/sailor/)**

<!-- ![alt text](https://github.com/UT-Austin-RPL/sailor/blob/web/src/overview.png) -->
<a href="https://ut-austin-rpl.github.io/sailor/" target="_blank"><img src="https://github.com/UT-Austin-RPL/sailor/blob/web/src/overview.png" width="90%" /></a>

## Installation
### Prerequisite: Setup robosuite 
1. Download [MuJoCo 2.0](https://www.roboti.us/download.html) (Linux and Mac OS X) and unzip its contents into `~/.mujoco/mujoco200`, and copy your MuJoCo license key `~/.mujoco/mjkey.txt`. You can obtain a license key from [here](https://www.roboti.us/license.html).
2. (linux) Setup additional dependencies: ```sudo apt install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common net-tools xpra xserver-xorg-dev libglfw3-dev patchelf```
3. Add MuJoCo to library paths: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin`

### Setup SAILOR codebase
1. Download code: ```git clone https://github.com/UT-Austin-RPL/sailor```
2. Create the conda environment: `cd sailor; conda env create --name sailor --file=sailor.yml`
2. (if above fails) edit `sailor.yml` to modify dependencies and then resume setup: `conda env update --name sailor --file=sailor.yml`
3. Activate the conda environment: `conda activate sailor`
4. Finish setup: `pip install -e .`

### Setup CALVIN environment
1. Download code: ```git clone --recurse-submodules https://github.com/snasiriany/calvin.git```
2. Setup tacto: ```cd calvin/calvin_env/tacto; pip install -e .```
3. Setup calvin_env: ```cd ..; pip install -e .```
4. Download all [calvin datasets](https://utexas.box.com/s/uo4vrbdmkrk8pt0j8sj2jt2vqhpnr2lx) and store under `{sailor_path}/datasets/calvin` directory

### Setup Franka kitchen environment
1. Download code (use the `sailor` branch): ```git clone -b sailor https://github.com/snasiriany/d4rl```
2. Setup d4rl: ```cd d4rl; pip install -e .```
3. Download all [Franka kitchen datasets](https://utexas.box.com/s/goxdssyxypd76vv2ezscomz0k2rnu3ub) and store under `{sailor_path}/datasets/franka_kitchen` directory

## Running SAILOR

### Step 1: pre-training
Generate the configs for pre-training the skill model: ```python robomimic/scripts/config_gen/sailor_pt.py --env {calvin or kitchen}```

Note: you can add the ```--debug``` flag for a small test run. Run the resulting command that is printed.
### Step 2: target task learning
Once the previous stage finishes, add the skill model checkpoints in `robomimic/scripts/config_gen/sailor_ft.py` under the `algo.policy.skill_params.model_ckpt_path` entry.

Then generate the configs for target task learning: ```python robomimic/scripts/config_gen/sailor_ft.py --env {calvin or kitchen}```

Note: you can add the ```--debug``` flag for a small test run. Run the resulting command that is printed.
## Citation
```bibtex
@inproceedings{nasiriany2022sailor,
  title={Learning and Retrieval from Prior Data for Skill-based Imitation Learning},
  author={Soroush Nasiriany and Tian Gao and Ajay Mandlekar and Yuke Zhu},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```
