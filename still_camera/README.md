# RGBD Snapshot App

A desktop application for capturing RGB and depth frames from an Intel RealSense D405 camera.

## Setup

1.  **Prerequisites:**
    - A conda distribution (like Miniconda or Anaconda).
    - Python 3.12
    - Intel RealSense SDK 2.0: Must be installed for `pyrealsense2` to work. Follow the official installation guide for your OS. On Linux, ensure udev rules are correctly set up.

2.  **Create and activate a conda environment:**
    
    *Note: Your shell must be configured to use `conda activate`. If you have issues, run `conda init` for your shell (e.g., `conda init bash`). You must then restart your terminal for the changes to take effect. If `conda init` reports "No action taken", your configuration is already correct, but you still need to restart your shell session.*
    
    ```bash
    conda create -n aider python=3.12 -y
    conda activate aider
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the App

(To be implemented)

```bash
conda run -n aider python -m rgbd_app.main
```

## How to Run Tests

```bash
conda run -n aider python -m unittest discover -s tests
```
