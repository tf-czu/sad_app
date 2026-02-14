# RGBD Snapshot App (Phase 1: RealSense D405)

## Goal
Create a desktop application (GUI) for configuring an RGBD camera and capturing:
- a single RGB + depth frame, or
- a sequence of frames with a fixed time interval.

Phase 1 targets **Intel RealSense D405 only**.
- Connected device
- Serial Number: **218722270084**
- Python version: **3.12**
- GUI toolkit: **Tkinter**
- Testing framework: **unittest**

Development happens in the directory where `aider` is launched.

---

## Constraints / Workflow

- Use **aider** for implementation.
- Use a **Ralph loop** (this file is re-used each iteration).
- Use a dedicated git branch:
  `feature/rgbd-snapshot-realsense`
- Commit frequently.
- **All commit messages and code comments MUST be in English.**
- Use clean architecture:
  - backend (camera + capture)
  - GUI (Tkinter)
  - IO
- Keep iterations small.

---

## Success Criteria (Phase 1)

1. Connect to RealSense D405 by serial number.
2. Live preview:
   - RGB stream
   - Depth stream (colorized for display)
3. Camera controls (if supported):
   - Exposure (manual)
   - Gain (ISO-equivalent)
   - Auto exposure toggle
   - Focus / Auto-focus if supported
4. Capture:
   - Single snapshot
   - Sequence capture (N frames, interval in ms)
5. Storage:
   - RGB: PNG
   - Depth: 16-bit PNG
   - Metadata JSON:
     - timestamp
     - serial
     - intrinsics
     - exposure/gain settings
     - frame index (for sequence)
6. Clean start/stop streaming.
7. Proper error messages if device not found.

---

## Non-goals (Phase 1)

- OAK camera support
- Multi-camera support
- Calibration GUI

---

## Suggested Stack

- Python 3.12
- pyrealsense2
- OpenCV
- numpy
- Tkinter (standard library)
- Pillow (for Tkinter image display)
- unittest

If something is unavailable, choose closest alternative and document it.

---

# Project Structure (Target)

rgbd_app/
    __init__.py
    main.py
    gui.py
    realsense_device.py
    capture_io.py
    capture_service.py
    utils.py

tests/
    test_capture_io.py
    test_metadata.py

scripts/
    realsense_smoke.py

captures/   (gitignored)

---

# Milestones

## Milestone 0 — Repo Setup

- [x] Ensure git repo initialized.
- [x] Create branch `feature/rgbd-snapshot-realsense`.
- [x] Add `.gitignore`:
      - __pycache__/
      - captures/
      - *.pyc
- [x] Add `requirements.txt`:
      pyrealsense2
      opencv-python
      numpy
      pillow
- [x] Add README.md with:
      - setup instructions
      - RealSense dependency notes
      - how to run app
      - how to run tests
- [x] Add `.aider.conf.yml`
      auto-commits: true
      commit-language: en

---

## Milestone 1 — RealSense Backend (No GUI)

- [x] Implement `realsense_device.py`:
      - list_devices()
      - connect(serial)
      - start()
      - stop()
      - get_frames()
      - get_intrinsics()
- [x] Convert:
      - RGB → numpy BGR (OpenCV)
      - Depth → numpy uint16
      - Depth visualization (colormap)
- [x] Add `scripts/realsense_smoke.py`
      - Connect by serial
      - Grab 30 frames
      - Print intrinsics
      - Exit cleanly

---

## Milestone 2 — Capture & Storage

- [x] Define capture directory structure:
      captures/YYYYMMDD_HHMMSS_SERIAL/
- [x] Implement `capture_io.py`:
      - save_rgb()
      - save_depth_16bit()
      - save_depth_preview()
      - save_metadata()
- [x] Implement `capture_service.py`:
      - capture_one()
      - capture_sequence(n, interval_ms)
- [x] Ensure sequence timing uses monotonic clock.

---

## Milestone 3 — Tkinter GUI v1 (Preview)

- [x] Create Tkinter main window.
- [x] Serial input field (default prefilled).
- [x] Connect / Disconnect buttons.
- [x] Two preview panels:
      - RGB
      - Depth (colorized)
- [x] Update preview in loop using after().
- [x] Status bar + simple logging text box.

---

## Milestone 4 — Camera Controls

- [x] Add:
      - Exposure slider/input
      - Gain slider/input
      - Auto exposure checkbox
- [x] Dynamically enable/disable controls if unsupported.
- [x] Apply settings live.

---

## Milestone 5 — Capture UI

- [x] Add "Capture Snapshot" button.
- [x] Add sequence controls:
      - N frames input
      - Interval (ms)
      - Start button
- [x] Run capture in background thread.
- [x] Show progress indicator.
- [x] Keep UI responsive.

---

## Milestone 6 — Testing & Quality

- [x] Add unittest for:
      - directory naming
      - metadata generation
      - file saving logic (mocked)
- [x] Ensure tests run via:
      python -m unittest discover
- [x] Ensure no camera is required for tests.

---

# Ralph Iteration Rules

Each iteration:

1. Read this file.
2. Pick the next small unchecked task(s).
3. Implement cleanly with English comments.
4. Run:
       python -m unittest discover
5. Fix issues if needed.
6. Commit changes (English commit message).
7. Update checkboxes.

Stop when all milestones are completed and verified with the connected D405 device.
