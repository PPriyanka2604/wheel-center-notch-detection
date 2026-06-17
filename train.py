from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")   # nano version

# Train
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,          # GPU
    project="runs",
    name="wheel_center_notch_v8"
)

# Validate
metrics = model.val()

print(metrics)