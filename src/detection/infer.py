import cv2
from ultralytics import YOLO
from config import MODEL_PATH, CAMERA_SOURCE, CONF_THRESHOLD

def run_inference():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Kato - Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()