import os
import json
import cv2
import numpy as np

from tree_monitor.model.detector import Detector


def process_images_to_annotations(images_folder, output_json, models_path):
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg')
    annotations_dict = {}
    canopy_detector = Detector(os.path.join(models_path, "best_seg.pt"))
    os.makedirs(os.path.join(images_folder, "tmp_cca"), exist_ok=False)

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

        detections = canopy_detector.detect(img)
        polygons = [poly for __, poly, __ in detections]

        # 2. Draw all polygons onto a blank binary mask to perform CCA
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(binary_mask, polygons, -1, color=255, thickness=cv2.FILLED)

        # 3. Calculate the border threshold (1% of the image width)
        border_threshold = int(round(width * 0.01))

        # Define the boundary limits for the top and bottom edges
        top_limit = border_threshold
        bottom_limit = height - border_threshold

        # 4. Perform Connected Component Analysis (CCA)
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

            # 5. Extract external contours of the connected tree crown
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
        annotations_dict[filename] = canopy_annotations

    # 6. Write the final results into a JSON file
    with open(os.path.join(images_folder, output_json), 'w', encoding='utf-8') as json_file:
        json.dump(annotations_dict, json_file, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='path to image dir')
    parser.add_argument('--models', help='Path to models', default="tree_monitor/model/my_models/medium/")
    args = parser.parse_args()

    process_images_to_annotations(args.images,"annotations_cca.json", args.models)
