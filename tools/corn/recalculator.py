import os
import pandas as pd

os.makedirs("results", exist_ok=True)

# Parameters
H = 0.5      # High of the camera in meters
F = 1141.2     # Focal length in pixels, obtained from camera calibration (camera-calibrator.py)

# Load clicks data
df = pd.read_csv("results/clicks-alltogether-noduplicates.csv")

# Calculate "x offset in meters" using the formula: x_offset_meters = (H * x_offset) / F
df["x_offset_meters"] = round((H * df["x_offset"]) / F, 4)

# Save deviations to CSV
df[["frame_number", "x_offset", "x_offset_meters"]].to_csv("results/deviations.csv", index=False)
