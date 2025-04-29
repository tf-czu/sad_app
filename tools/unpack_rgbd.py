import sys

import numpy as np
import cv2

rgbd_file = sys.argv[1]
data = np.load(rgbd_file)
color = data["img"]
depth = data["depth"]

# save color image
cv2.imwrite(rgbd_file.replace(".npz", "_color.png"), color)

depth[depth>4000] = 0
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_8bit = depth_normalized.astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

# save color image
cv2.imwrite(rgbd_file.replace(".npz", "_depth.png"), depth_colored)
