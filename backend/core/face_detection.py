import cv2
import os

class FaceDetector:
    def __init__(self):
        # Load Haar Cascade from the models folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        haarcascade_path = os.path.join(
            base_dir,
            "models",
            "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

        if self.face_cascade.empty():
            raise FileNotFoundError(
                f"Face detection model not found: {haarcascade_path}"
            )

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        detections = []

        for (x, y, w, h) in faces:
            detections.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "score": 1.0
            })

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return frame, detections
