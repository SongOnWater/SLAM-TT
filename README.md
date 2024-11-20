# SLAM-TT: Simultaneous Localization and Mapping for Table Tennis

Original TTNet Repo: https://github.com/AugustRushG/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch

WHAM Repo: https://github.com/yohanshin/WHAM

---


## Setup Guide

### Prerequisites

To ensure smooth setup and functionality, make sure your system meets the following requirements:
- Operating System: Linux or Windows Subsystem for Linux (WSL)
- Graphics: Nvidia GPU with CUDA support.

### Overview

TTNet: Used to detect ball position and bounce
WHAM: Used to detect player movements
WHAM_TO_BLENDER: Used to take WHAM ```.pkl``` output and export it to blender
Unity: Used to render the entire scene

### 1. TTNet

Full Instructions [Here](TTNet/README.md)

```bash
conda create -n ttnet python=3.9
conda activate ttnet

pip install -U -r requirement.txt

sudo apt-get install libturbojpeg
pip install PyTurboJPEG

# WSL Users: fix cv2.imshow()
sudo apt-get install libgl1-mesa-glx
sudo apt-get install xdg-utils
```

### 2. WHAM

Please see [Installation](WHAM/docs/INSTALL.md) for details.

Usage:
```
python demo.py --video examples/IMG_9732.mov --visualize --save_pkl
```

### 3. WHAM TO BLENDER

Full Instructions: https://youtu.be/7heJSFGzxAI?si=8c1HD1Ux81eDpkLu&t=380
Note that I modified the script to be compatible with Blender>=4.1, so use the files I provide in this repo

1. Register an account and download smpl model for Maya. (Hint: Select "Download version 1.0.2 for Maya"): https://smpl.is.tue.mpg.de/download.php
2. Open WHAM_TO_BLENDER.blend
3. Select ```Joblib Install``` in the script selection menu and press the Play button (Note: this will install joblib to your global pip packages)
4. Select ```FINAL_Script```
5. Modify these lines:
```python
character = 0 # There are two players. Choose the index to focus on
pre_date = r"\\wsl.localhost\Ubuntu\home\dylan\Coding\SLAM-TT\WHAM\output\demo\test_1_trimmed\wham_output.pkl" # Set this to your .pkl output path from the previous step
packages_path= r"c:\users\dylan\appdata\roaming\python\python311\site-packages" # Add your python packages to the path (wherever you installed joblib)
```
6. Press the Play button
7. Set the x rotation to -90deg

### 4. Move Everything to Unity

TODO: