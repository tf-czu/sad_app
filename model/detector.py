"""
    Tree detector
"""

import cv2
import numpy as np
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
        :return: list of detections [(bbox, polygon, score)
        """
        results = self.model(image, verbose=False)
        detections = []

        for result in results:
            if result.masks is not None:
                for box, mask in zip(result.boxes, result.masks.xy):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf.item())
                    polygon = np.array(mask, np.int32)
                    detections.append(((x1, y1, x2, y2), polygon, score))
            else:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf.item())
                    detections.append(((x1, y1, x2, y2), None, score))

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('img', help='Path to image')
    parser.add_argument('--model', help='Path to model', default="my_models/run4_best_m.pt")
    args = parser.parse_args()

    detector = Detector(model_path=args.model)
    image = cv2.imread(args.img)
    detections = detector.detect(image)

    for (x1, y1, x2, y2), polygon, conf in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if polygon is not None:
            # cv2.polylines(image, [polygon], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.drawContours(image, [polygon], -1, color=(0, 0, 255), thickness=-1)
        label = f"{conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
