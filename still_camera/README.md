# RGBD Snapshot App

A desktop application for capturing RGB and depth frames from Intel RealSense
(D405/D435) and Luxonis OAK cameras (including the OAK-D-Pro PoE, model "A",
where the color sensor is a separate camera from the mono pair).

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

## Camera Configuration (cameras.json)

Each camera is described by an entry in `cameras.json`:

```json
{
    "name": "OAK-D-poe",
    "type": "oak",
    "serial": "1844301001F17B0E00",
    "model": "A",
    "width": 1920,
    "height": 1080,
    "fps": 10
}
```

- `type`: `realsense` or `oak`.
- `model` (OAK only, optional): hardware layout identifier.
  - `"A"` – separate color (RGB) sensor on `CAM_A`, mono pair on `CAM_B`/`CAM_C`
    (e.g. OAK-D-Pro / OAK-D-POE). **Required for OAK-D-POE.**
  - `"SR"` – color and left mono share the sensor on `CAM_B`, right mono on `CAM_C`
    (e.g. OAK-D-SR).
  - When `model` is omitted, it is auto-detected from the device features.

## How to Run the App

(To be implemented)

```bash
conda run -n aider python -m rgbd_app.main
```

## How to Run Tests

```bash
conda run -n aider python -m unittest discover -s tests
```
