from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
results = model.train(data='data_seg.yaml', epochs=200, imgsz=640)
