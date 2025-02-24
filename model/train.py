from ultralytics import YOLO

model = YOLO('yolov8x.pt')
results = model.train(data='data.yaml', epochs=200, imgsz=640)
