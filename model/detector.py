"""
    Tree detector
"""

import cv2
import torch
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, device=None):
        """
        Initialization YOLO.
        :param model_path: path to model (.pt file).
        :param device: 'cuda' or 'cpu' (auto in no mentioned).
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        print(f"YOLO model loaded to {self.device}")

    def detect(self, image):
        """
        Detects objects from image.
        :param image: Input image (BGR).
        :return: list of detections [(bbox, score)
        """
        results = self.model(image)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0].item())
                detections.append(((x1, y1, x2, y2), score))

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('img', help='Path to image')
    parser.add_argument('--model', help='Path to model', default="run4_best.pt")
    args = parser.parse_args()

    detector = Detector(model_path=args.model)
    image = cv2.imread(args.img)
    detections = detector.detect(image)

    for (x1, y1, x2, y2), conf in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
