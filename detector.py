
import cv2
from ultralytics import YOLO


def load_model():
    """Load the pretrained YOLO model."""
    return YOLO("yolo11n.pt")


def detect_people(model, image, confidence=0.40):
    """Detect people and return count, annotated image, and details."""

    results = model.predict(
        source=image,
        classes=[0],
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Convert the annotated image from BGR to RGB
    annotated_image = cv2.cvtColor(
        result.plot(),
        cv2.COLOR_BGR2RGB
    )

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "person_id": len(detections) + 1,
                "confidence": float(box.conf[0]),
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2)
            })

    return len(detections), annotated_image, detections
