import os
import json
import cv2
import numpy as np

from tree_monitor.model.detector import Detector


def process_images_to_annotations(images_folder, output_json, models_path):
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg')
    bbox_annotations = {}
    contour_annotations = {}
    cca_annotations = {}
    tree_detector = Detector(os.path.join(models_path, "best.pt"))
    canopy_detector = Detector(os.path.join(models_path, "best_seg.pt"))
    os.makedirs(os.path.join(images_folder, "tmp_cca"), exist_ok=False)
    os.makedirs(os.path.join(images_folder, "tmp_bbox"), exist_ok=False)
    os.makedirs(os.path.join(images_folder, "tmp_contours"), exist_ok=False)

    for filename in sorted(os.listdir(images_folder)):
        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(images_folder, filename)
        print(f"Processing: {filename}")

        # Get image dimensions to initialize the blank mask
        # We read in grayscale mode just to quickly get height and width
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue
        height, width = img.shape[:2]
        assert height == 1080 and width == 1920, (height, width)

        # Calculate the border threshold (1% of the image width)
        border_threshold = int(round(width * 0.01))

        # Define the boundary limits for the top and bottom edges
        top_limit = border_threshold
        bottom_limit = height - border_threshold

        # 1. Bbox detections from tree_detector (best.pt)
        tree_detections = tree_detector.detect(img)
        bbox_list = []
        bbox_debug_img = img.copy()
        for (x1, y1, x2, y2), __, __ in tree_detections:
            # Draw every detected bbox in green first
            cv2.rectangle(bbox_debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if y1 >= top_limit and y2 <= bottom_limit:
                bbox_list.append([x1, y1, x2, y2])
                # Redraw the bbox in red if it passed the boundary check
                cv2.rectangle(bbox_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        bbox_annotations[filename] = bbox_list
        cv2.imwrite(os.path.join(images_folder, "tmp_bbox", f"check_{filename}"), bbox_debug_img)

        # 2. Raw contour detections from canopy_detector (best_seg.pt) without CCA
        canopy_detections = canopy_detector.detect(img)
        raw_polygons = [poly for __, poly, __ in canopy_detections]
        contour_list = []
        contour_debug_img = img.copy()
        for polygon in raw_polygons:
            # Draw every detected polygon in green first
            cv2.drawContours(contour_debug_img, [polygon], -1, (0, 255, 0), 2)

            # Boundary check: skip the polygon if even a single point
            # falls within the 1% width zone of the top or bottom edge
            if np.any(polygon[:, 1] < top_limit) or np.any(polygon[:, 1] > bottom_limit):
                continue  # Skip this crown

            # Convert to match the target nested list JSON structure:
            # [[[[x1, y1]], [[x2, y2]], ...]]
            formatted_contour = polygon.reshape(-1, 1, 2).tolist()
            contour_list.append([formatted_contour])
            # Redraw the polygon in red if it passed the boundary check
            cv2.drawContours(contour_debug_img, [polygon], -1, (0, 0, 255), 2)
        contour_annotations[filename] = contour_list
        cv2.imwrite(os.path.join(images_folder, "tmp_contours", f"check_{filename}"), contour_debug_img)

        # 3. CCA processing (existing logic)
        polygons = [poly for __, poly, __ in canopy_detections]

        # Draw all polygons onto a blank binary mask to perform CCA
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(binary_mask, polygons, -1, color=255, thickness=cv2.FILLED)

        # Perform Connected Component Analysis (CCA)
        # Using 8-connectivity to group pixels touching diagonally as well
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        canopy_annotations = []
        debug_img = img.copy()
        # Iterate through all found components (label 0 is always the background)
        for label in range(1, num_labels):
            # Create a binary mask isolated for the current component
            component_mask = (labels_im == label).astype(np.uint8) * 255

            # Find coordinates [y, x] of all pixels belonging to this component
            pixel_y, pixel_x = np.where(labels_im == label)

            # Boundary check: exclude the component if even a single pixel
            # falls within the 1% width zone of the top or bottom edge
            if np.any(pixel_y < top_limit) or np.any(pixel_y > bottom_limit):
                continue  # Skip this crown

            # Extract external contours of the connected tree crown
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            canopy_list = []
            cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(images_folder, "tmp_cca", f"check_{filename}"), debug_img)

            for contour in contours:
                # OpenCV contours have a default shape of (N, 1, 2) containing [X, Y]
                # Convert it to match your target nested list JSON structure:
                # [[[[x1, y1]], [[x2, y2]], ...]]
                formatted_contour = contour.tolist()
                canopy_list.append(formatted_contour)
            canopy_annotations.append(canopy_list)

        # Store the collected crown annotations for the current image
        cca_annotations[filename] = canopy_annotations

    # Write the final results into JSON files
    with open(os.path.join(images_folder, "annotations_bbox.json"), 'w', encoding='utf-8') as json_file:
        json.dump(bbox_annotations, json_file, indent=4)
    with open(os.path.join(images_folder, "annotations_contours.json"), 'w', encoding='utf-8') as json_file:
        json.dump(contour_annotations, json_file, indent=4)
    with open(os.path.join(images_folder, output_json), 'w', encoding='utf-8') as json_file:
        json.dump(cca_annotations, json_file, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='path to image dir')
    parser.add_argument('--models', help='Path to models', default="tree_monitor/model/my_models/medium/")
    args = parser.parse_args()

    process_images_to_annotations(args.images, "annotations_cca.json", args.models)