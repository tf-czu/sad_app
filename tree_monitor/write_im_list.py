import csv
import os
import sys


def generate_image_csv(root_directory):
    # Valid extensions for JPEG images
    valid_extensions = (".jpg", ".jpeg", ".JPG", ".JPEG")

    # Walk through the root directory
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Filter files to include only JPEG images
        image_files = [f for f in filenames if f.endswith(valid_extensions)]

        # If the directory contains at least one JPEG, create the CSV file
        if image_files:
            csv_file_path = os.path.join(dirpath, "images_list.csv")

            try:
                with open(
                    csv_file_path, mode="w", newline="", encoding="utf-8"
                ) as csv_file:
                    writer = csv.writer(csv_file)

                    # Write the specified header
                    writer.writerow(["im_name", "tree1", "tree2"])

                    # Write image names; columns tree1 and tree2 are left empty
                    for img in sorted(image_files):
                        writer.writerow([img, "", ""])

                print(f"Successfully created CSV in: {dirpath}")
            except Exception as e:
                print(f"Error creating CSV in {dirpath}: {e}")


if __name__ == "__main__":
    # Define the target directory path
    target_directory = sys.argv[1]

    if os.path.exists(target_directory):
        generate_image_csv(target_directory)
    else:
        print(f"Directory '{target_directory}' does not exist.")