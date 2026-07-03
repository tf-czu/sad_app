import sys
import os
import csv
from ultralytics import YOLO

assert len(sys.argv) == 3
model_path = sys.argv[1]
data_path = sys.argv[2]

model = YOLO(model_path)
metrics = model.val(data=data_path, imgsz=640, split='test', project='evaluation', save_json=False, plots=True)

precision, = metrics.box.p_curve
recall, = metrics.box.r_curve

pr_data_path, __ = os.path.split(model_path)
with open(os.path.join(pr_data_path, "pr_data.csv"), 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=",")
    csv_writer.writerow(["Precision", "Recall"])
    for items in zip(precision, recall):
        csv_writer.writerow(items)
