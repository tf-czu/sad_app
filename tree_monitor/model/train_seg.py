from ultralytics import YOLO

model = YOLO('yolov11n-seg.pt')
results = model.train(data='data_seg.yaml', epochs=200, imgsz=640)
