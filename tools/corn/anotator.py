import os
os.environ["QT_LOGGING_RULES"] = "*=false"
import csv
import cv2  # type: ignore

# Folders and paths
VIDEO_PATH = "videos/corn.mp4"
RESULTS_DIR = "results"
ANNOTATED_DIR = os.path.join(RESULTS_DIR, "anotated-frames")
METADATA_PATH = os.path.join(RESULTS_DIR, "metadata.txt")
CSV_PATH = os.path.join(RESULTS_DIR, "clicks.csv")

os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    raise Exception("Video se nepodařilo otevřít.")

# Get video metadata
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Save metadata
with open(METADATA_PATH, "w") as f:
    f.write(f"FPS: {fps}\n")
    f.write(f"Total frames: {total_frames}\n")

# Prepare CSV file for clicks
csv_file = open(CSV_PATH, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_number",
    "time_s",
    "x",
    "y",
    "x_offset"
])

# Initialization
current_frame_idx = 0
current_frame = None

# Load function, to load a specific frame
def load_frame(frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# Mouse callback function to capture clicks
def mouse_callback(event, x, y, flags, param):
    global current_frame_idx, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # Load current frame to make copy
        if current_frame is None:
            return
        frame = current_frame.copy()
        # Calculate time and offset
        time_sec = current_frame_idx / fps
        height, width = frame.shape[:2]
        center_x = round(width / 2, 2)
        x_offset = x - center_x
        # Save click data to CSV
        csv_writer.writerow([
            current_frame_idx + 1,
            round(time_sec, 2),
            x,
            y,
            round(x_offset, 2)
        ])
        csv_file.flush()
        # Draw annotation
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1) # Draw a circle at click position -> params: image, center of the circle, radius, color, thickness (-1 for filled circle)
        info_text = f"Frame: {current_frame_idx + 1}/{total_frames} | Time: {current_frame_idx/fps:.2f}s"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA) # Put text in image -> params: image, text, position, font, size, color, thickness, parameter for smoothness
        # Refresh window immediately
        cv2.imshow("Corn video", frame)
        # Save frame
        output_path = os.path.join(
            ANNOTATED_DIR,
            f"frame_{current_frame_idx + 1}.png"
        )
        cv2.imwrite(output_path, frame) # Save annotated image

# Set a name for the window (to reference it later) and register mouse callback
cv2.namedWindow("Corn video")
cv2.setMouseCallback("Corn video", mouse_callback)

# Main loop
while True:
    # Load wanted frame
    current_frame = load_frame(current_frame_idx)
    if current_frame is None:
        break
    frame = current_frame.copy()
    # Render metadata info
    info_text = f"Frame: {current_frame_idx + 1}/{total_frames} | Time: {current_frame_idx/fps:.2f}s"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Corn video", frame) # Show image in window with given name -> params: window name, image
    # Wait for key press
    key = cv2.waitKey(0)
    # Handle key presses
    if key == ord('q'):
        break
    elif key == ord('d'):        # Next frame
        current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
    elif key == ord('a'):        # Previous frame
        current_frame_idx = max(current_frame_idx - 1, 0)
    elif key == ord('f'):        # Jump +15 frames
        current_frame_idx = min(current_frame_idx + 15, total_frames - 1)
    elif key == ord('b'):        # Jump -15 frames
        current_frame_idx = max(current_frame_idx - 15, 0)
    elif key == ord('p'):        # Jump +500 frames
        current_frame_idx = min(current_frame_idx + 500, total_frames - 1)

# Free resources
csv_file.close()
cap.release()
cv2.destroyAllWindows()
