# SSFold: A <u>S</u>ystem for <u>S</u>tructured <u>F</u>olding of Arbitrary Crumpled Cloth Using Graph Dynamics from Human Demonstration

## Fold cloth from human demonstrations
![Overview](./images/fig1.png)

## Method Overview
![Overview](./images/fig2.png)

## Introduction
Robotic cloth manipulation faces challenges due to the fabric’s complex dynamics and the high dimensionality of configuration spaces. Previous methods have largely focused on isolated smoothing or folding tasks and overly reliant on simulations, often failing to bridge the significant sim-to-real gap in deformable object manipulation. To overcome these challenges, we propose a two-stream architecture with sequential and spatial pathways, unifying smoothing and folding tasks into a single adaptable policy model that accommodates various cloth types and states. The sequential stream determines the pick and place positions for the cloth, while the spatial stream, using a connectivity dynamics model, constructs a visibility graph from partial point cloud data of the self-occluded cloth, allowing the robot to infer the cloth’s full configuration from incomplete observations. To bridge the sim-to-real gap, we utilize a hand tracking detection algorithm to gather and integrate human demonstration data into our novel end-to-end neural network, improving real-world adaptability.

## Installation
1.Clone this repository

2.install some packages

  ~~~
    * pytorch, torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
    * torch-scatter, torch-sparse, torch-geometric (related to the version of pytorch)
    * einops: `pip install einops`
    * tqdm: `pip install tqdm`
    * yaml: `pip install PyYaml`
  ~~~


## Structure of the Repository
This repository provides data and code as follows.
```
    calibration/             # Robotic arm calibration and desktop calibration
    cam/                     # TCamera acquisition program for experiments
    models/                   # Experimental model structure
    robots/                  # Detailed documentation of the UR5 robot arm used in the real world
    software/                # TSoftware installation package
    tools/          # Tools in the experiment
    util/          # Visualization and some structure of the model
```
You can follow the README in EACH FOLDER to install and run the code.

## Running SSFold in the real world

### Camera calibration
To find and grasp the 3D position in the real world setup of a chosen pixel in the RGB-D image, the camera pose relative to the arms' bases can be calibrated with

```sh
python calibration_table.py.py
python calibration_robot.py
```
The output of this script are a depth scaling and the relative poses of the camera to the right arm and left arm, saved as `camera_depth_scale.txt`,  `top_down_ur5_cam_pose.txt`, respectively.


r

# SSFold-main
