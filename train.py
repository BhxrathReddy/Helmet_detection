from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data ="helmet.yaml", imgsz = 640,
            batch = 8 ,epochs = 100 , workers = 0 ,device = 0)