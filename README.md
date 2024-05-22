# tbai
Towards better athletic intelligence

## Implemented controllers

```
📦tbai
 ┣ 📂tbai_static               # Static (high gain PD) controller
 ┣ 📂tbai_mpc_perceptive       # Perceptive NMPC controller [1]
 ┣ 📂tbai_mpc_blind            # Blind NMPC controller [1]
 ┣ 📂tbai_rl_perceptive        # Perceptive RL controller [2]
 ┣ 📂tbai_rl_blind             # Blind RL controller [2]
 ┣ 📂tbai_dtc                  # DTC controller (perceptive) [3]
 ┣ 📂tbai_joe                  # Perceptive NMPC controller with NN-based tracking controller [1],[3]

 [1] Perceptive Locomotion through Nonlinear Model Predictive Control
     https://arxiv.org/abs/2208.08373
 [2] Learning robust perceptive locomotion for quadrupedal robots in the wild
     https://arxiv.org/abs/2201.08117
 [3] DTC: Deep Tracking Control
     https://arxiv.org/abs/2309.15462
```

## Perceptive MPC


https://github.com/lnotspotl/tbai/assets/82883398/ced1724a-73b3-4aa4-86bd-f9dfa2160ca8


## Blind MPC


https://github.com/lnotspotl/tbai/assets/82883398/f1ce4b7e-87ed-49a7-acb7-9fe580bbdcea


## Perceptive RL


https://github.com/lnotspotl/tbai/assets/82883398/ff93c5c5-3129-459f-b1b4-30c9cb928301


## Blind RL


https://github.com/lnotspotl/tbai/assets/82883398/43354e1b-b451-4fed-a09a-af8a49fe80d0


## Perceptive DTC

## Blind DTC

## System architecture

![architecture](https://github.com/lnotspotl/tbai/assets/82883398/3a21ead9-75dd-4e27-9a8c-c59526a45ae5)

## Installing libtorch C++
There are two steps to installing `libtorch`. First, you need to download a suitable `libtorch` version.
Once the library is downloaded, it's necessary to create a symlink to it in the `dependencies` folder.
Here's how to do it:

### 1. Getting libtorch download link
Get your download link from the [official PyTorch website](https://pytorch.org/). Note that opting for the `(cxx11 ABI)` version is paramount.
If you download the `(Pre-cxx11 ABI)` version, things won't work as necessary.


![image](https://github.com/lnotspotl/tbai/assets/82883398/183255fc-83c5-4bab-a48d-f70e5c7593d7)


### 2. Downloading libtorch and creating a symlink
Now that you have your url, you can download the library, unzip it and create a symlink in the `dependencies` folder.
```bash
wget <your-url>
unzip <downloaded-zip> -d <your-folder>  # can be `dependencies`
ln -s <your-folder>/libtorch dependencies  # Only necessary if, in the previous step, you did not unzip in `dependencies`
```
Your `dependencies` folder should not look as follows:
<p align="center">
  <img src="https://github.com/lnotspotl/tbai/assets/82883398/657d8681-1abd-4dae-b4c2-15347ed542fd" />
</p>
That's it. You should now be able to compile the entire project. Enjoy 🤗

## Credits
This project stands on the shoulders of giants.
None of this would have been possible were it not for many amazing open-source projects.
Here are a couple that most inspiration was drawn from and that were instrumental during the development:

- https://github.com/leggedrobotics/ocs2
- https://github.com/qiayuanl/legged_control
- https://github.com/leggedrobotics/legged_gym
- https://github.com/leggedrobotics/rsl_rl
- https://github.com/ANYbotics/elevation_mapping
- https://github.com/leggedrobotics/elevation_mapping_cupy
- https://github.com/bernhardpg/quadruped_locomotion
- https://github.com/stack-of-tasks/pinocchio
- https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
- https://github.com/mayataka/robotoc
- https://github.com/mayataka/legged_state_estimator
- hundreds of others ...

Thank you all 🤗
