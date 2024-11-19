# SLAM-TT

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

TTNet Repo: https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch

---


## Setup Guide

### 1. TTNet

TODO:

### 2. WHAM

TODO: 

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