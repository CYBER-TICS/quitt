from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs",
        name="kato_train"
    )

if __name__ == "__main__":
    train()