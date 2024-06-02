<center> <h1> FaceGSM </h1> </center>

# Table of Contents <!-- omit from toc -->

-   [FaceGSM](#facegsm)
-   [Installation](#installation)
    -   [Environment Setup and Dependencies Installation](#environment-setup-and-dependencies-installation)
        -   [\*Requirements for Windows Users](#requirements-for-windows-users)
        -   [1. Clone the Repository](#1-clone-the-repository)
        -   [2. Python Environment](#2-python-environment)
        -   [3. Installing Required Packages](#3-installing-required-packages)
-   [Features and Usage Guide](#features-and-usage-guide)
    -   [Manual Mode](#manual-mode)
    -   [Camera Capture Mode](#camera-capture-mode)
    -   [Live Camera Mode](#live-camera-mode)
        -   [Face Recognition Mode](#face-recognition-mode)
        -   [Attack Mode](#attack-mode)
        -   [Running Live Camera Mode](#running-live-camera-mode)
    -   [Caveats](#caveats)
    -   [Credits](#credits)

# FaceGSM

FaceGSM is an open source penetration testing tool that automates the FGSM adversarial attack on the FaceNet model. It comes with three main modes such as manual mode, live mode, and camera mode. Additionally, we have implement some features to improve FaceGSM's efficiency including checkpoints feature and database generator feature.

# Installation

## Environment Setup and Dependencies Installation

### \*Requirements for Windows Users

For `Windows` with GPU support, you can use WSL2 for TensorFlow GPU support. You can follow the instructions [here](https://www.tensorflow.org/install/pip#windows-wsl2). After succesfully installing WSL2, you can follow the same instructions below.

### 1. Clone the Repository

<!-- You can download FaceGSM by cloning the [FaceGSM]() repository: -->
_*disclaimer: product is still in development, please clone from dev branch by using this command_
```bash
git clone --single-branch --branch dev https://github.com/hahahohocorp/FaceGSM.git
```

### 2. Python Environment

Create a python environtment using [Python 3.10.11](https://www.python.org/downloads/) (recommended to use [Conda](https://docs.conda.io/en/latest/miniconda.html)). Alternatively, you can use the default Python environment on your system.

### 3. Installing Required Packages

After creating the environment, you can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

# Features and Usage Guide

## Quick Start
```bash
python3 facegsm.py manual --original original.jpg --target target.jpg --dataset /path/dataset/folder --checkpoint
```

To get a list of basic features and options use:

```bash
$ python3 facegsm.py --help

Usage: python3 facegsm.py live/camera/manual/--help
Options:
  live: Live camera feature in FaceGSM includes real-time face recognition and attack capabilities.
  camera: Camer original and target photos in FaceGSM.
  manual: Manual input for FGSM attack in FaceGSM.
  --help: Show help for available options.
```

FaceGSM has three main commands:

-   manual - Manual input for FGSM attack in FaceGSM.
-   camera - Camera capture for original and target photos in FaceGSM.
-   live - Live camera feature in FaceGSM includes real-time face recognition and attack capabilities.

By default, FaceGSM will utilize its own models, with a model output dimension of 160 x 160. However, user may use their own facial recognition model using `--models` options.

FaceGSM has its own datasets using 100 face from VGGFace datasets which picked randomly. User could modify these datasets by specifying their own dataset folder with `--datasets` options. FaceGSM then will create a JSON database file containing predictions output from the model.

Any output generated by FaceGSM will be stored in `output` folder.

Checkpoint can be used to accelerate the FGSM attack process if the user need to repeat the attack or target similar faces. User may use `--checkpoint` options to enable the checkpoint features.

## Manual Mode

Manual mode allows you to specify `original face` and `target face` in static images format such as `.jpg` and `.png`. You can use the following command to run the manual mode:

```bash
python3 facegsm.py manual --original ./path/to/original.png --target ./path/to/target.png
```

## Camera Capture Mode

Camera capture mode allows you to capture the `attacker original's face` and the `target victim's face` using `camera`. You can use the following command to run the camera capture mode:

```bash
python3 facegsm.py camera --host [ip_droidcam]

# Example:
# python3 facegsm.py camera --host http://192.168.1.3:4747/video
# python3 facegsm.py camera --host https://172.22.1.2:4343/video
```

_Disclaimer: This mode requires a third-party application called [DroidCam](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&pcampaignid=web_share) which can only be installed on Android devices. FaceGSM will connect to the camera wirelessly via the DroidCam IP address on the user's smartphone._

## Live Camera Mode

The live camera mode in FaceGSM enables real-time adversarial attack FGSM. In this mode, FaceGSM will have two sub-modes, namely `Face Recognition mode` and `Attack mode`.

### Face Recognition Mode

`Face Recognition Mode` is the default sub-mode when running live camera mode. This mode will perform face recognition using the live camera feed and compare the detected faces with those in the database. If a match is found, the program will generate a prediction of the face and its similarity value. These results are displayed alongside the live camera feed.

### Attack Mode

While the `Face Recognition Mode` is running, you can toogle the `Attack Mode` by pressing the `a` key from your keyboard. This mode will perform an targeted adversarial attack FGSM using the face detected by the live camera feed as the `Original Face` and the image path specified in the `--target` option as the `Target Face`.

### Running Live Camera Mode

To be able to run live camera mode, first you need to run `server.py`. Use the following command to run `server.py`:

```bash
python3 /app/server.py
# if the server is running correctly, you will see the following output:
Server started...
```

After the server is running, you can run FaceGSM `Live Camera Mode` using the following command:

```bash
python3 facegsm.py live --host [ip_droidcam] --target ./path/to/target.png
```

## Caveats

FaceGSM currently exclusively utilizes Facenet models, which limits potential vulnerabilities in facial recognition models susceptible to FGSM attacks.

## Credits

FaceGSM is developed by [Excy](), [Fejka](), and [Maskirovka]()