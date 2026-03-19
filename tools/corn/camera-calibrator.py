import cv2
import numpy as np

# Folders and paths
VIDEO_PATH = "videos/calibration.MOV"

# Calibration settings - number of inner corners, dimension of the checkerboard (being used for calibration)
CHECKERBOARD = (5, 3)

# Prepare matrix, rows according to the number of corners, columns = 3 = (x, y, z) coordinates, initially filled with zeros, data type = float32
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# Prepare the x, y coordinates of the checkerboard corners, z coordinate is 0 because we are using a flat checkerboard
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the frames
objpoints = []
imgpoints = []

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Init solved frame counter and used frames counter
frame_id = 0
used_frames = 0

# Loop through video frames
while True:
    # Read next frame
    ret, frame = cap.read()
    # If no frame is read, we have reached the end of the video
    if not ret:
        break
    # Increment frame counter
    frame_id += 1
    # We will analyze only every 7th frame, to speed up the process and avoid similar frames
    if frame_id % 7 != 0:
        continue
    # Convert frame to grayscale, because findChessboardCorners works with it better
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners, found is a boolean (whether corners were detected), corners are the coordinates of the corners
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
    # If corners are found, add object points and image points to the arrays
    if found:
        # Print analyzed frame id
        print("Analyzuji frame:", frame_id)
        # Add object points and image points to the arrays
        objpoints.append(objp) # Because the object points are the same for all frames, we can just add the same objp for each frame where corners are found
        imgpoints.append(corners) # Corners are the image points, they are different for each frame, so we add the corners detected in the current frame
        # Increment used frames counter
        used_frames += 1
        # Draw and display the corners detected in the current frame, to visually verify that they are correct
        # cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)
        # cv2.imshow("Corners in the frame", frame)
        # cv2.waitKey(200)

# Close video and all windows
cap.release()
cv2.destroyAllWindows()

# Print number of used frames
print("Použito framů:", used_frames)

# Calibrate the camera using the object points and image points, gray.shape[::-1] gives the size of the image (width, height), "None & None" means that we do not have initial estimates of results
# MTX gives us the camera parameters, RET gives us the error (the lower the better - usually between 0.1 and 1.0)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Get f_x and f_y from the camera matrix, they are the focal lengths (ohniskové vzdálenosti) in x and y direction
f_x = round(mtx[0, 0], 2)
f_y = round(mtx[1, 1], 2)

# F is the average of f_x and f_y, which is often used as a single focal length value for simplicity, when the camera has square pixels (f_x and f_y are similar)
f = round((f_x + f_y) / 2, 2)

# Print results
print("\nError:", ret)
print("F_x:", f_x)
print("F_y:", f_y)
print("F (průměr):", f)