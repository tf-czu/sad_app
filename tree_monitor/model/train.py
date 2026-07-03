from ultralytics import YOLO

model = YOLO('yolov11n.pt')
results = model.train(data='data.yaml', epochs=200, imgsz=640)
