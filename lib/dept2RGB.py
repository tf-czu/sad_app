import numpy as np
import cv2

def depth_to_rgb_aligned(depth_img, depth_intr, rgb_intr, T, rgb_shape):
    height_d, width_d = depth_img.shape
    height_rgb, width_rgb, __ = rgb_shape

    aligned_depth = np.zeros((height_rgb, width_rgb), dtype=np.uint16)

    fx_d, fy_d = depth_intr[0][0], depth_intr[1][1]
    cx_d, cy_d = depth_intr[0][2], depth_intr[1][2]

    fx_rgb, fy_rgb = rgb_intr[0][0], rgb_intr[1][1]
    cx_rgb, cy_rgb = rgb_intr[0][2], rgb_intr[1][2]

    for v in range(0, height_d, 1):
        for u in range(0, width_d, 1):
            z = depth_img[v, u] / 1000.0  # convert to m
            if z == 0:
                continue

            # Deprojection the depth pixel to 3D poit in depth frame
            x = (u - cx_d) * z / fx_d
            y = (v - cy_d) * z / fy_d
            point_d = np.array([x, y, z, 0])

            # Convert to RGB frame
            point_rgb = T @ point_d
            x_rgb, y_rgb, z_rgb, __ = point_rgb
            # x_rgb, y_rgb, z_rgb, __ = point_d  # if the transformation is skipped.

            if z_rgb <= 0:
                continue  # point is behind the camera

            # Projection to the RGB frame
            u_rgb = int((x_rgb * fx_rgb) / z_rgb + cx_rgb)
            v_rgb = int((y_rgb * fy_rgb) / z_rgb + cy_rgb)

            # Save depth if it is within the frame
            if 0 <= u_rgb < width_rgb and 0 <= v_rgb < height_rgb:
                current = aligned_depth[v_rgb, u_rgb]
                z_mm = int(z_rgb * 1000)

                if current == 0 or z_mm < current:
                    aligned_depth[v_rgb, u_rgb] = z_mm

    return aligned_depth

if __name__ == "__main__":
    # Got for oak-d-pro-poe 192.168.1.53, left to rgb camera.
    fc = 1551.5
    c_xc = 966.4
    c_yc = 542.9
    fd = 796.8
    c_xd = 644.3
    c_yd = 360.8

    color_intr = [[fc, 0, c_xc],
                  [0, fc, c_yc],
                  [0, 0, 1]]
    depth_intr = [[fd, 0, c_xd],
                  [0, fd, c_yd],
                  [0, 0, 1]]

    T_l = np.array([
        [9.99942660e-01, 1.45698769e-03, 1.06102219e-02, -3.73109245e+00/100],
        [-1.62577280e-03, 9.99872029e-01, 1.59166064e-02, - 1.44884959e-02/100],
        [-1.05856732e-02, -1.59329437e-02, 9.99817014e-01, -2.35053167e-01/100],
        [0, 0, 0, 1]
    ])

    T_r = np.array([
    [9.99974489e-01, -2.38180975e-03, 6.73444010e-03, 3.72088194e+00/100],
    [2.29958771e-03, 9.99923050e-01, 1.21906763e-02, -4.58627492e-02/100],
    [-6.76295767e-03, -1.21748783e-02, 9.99903023e-01, -2.50154287e-01/100],
    [0, 0, 0, 1]
    ])

    data = np.load("test_rgbd.npz")
    depth = data["depth"]
    print(depth.shape)
    color = data["img"]
    print(color.shape)
    aligned_depth = depth_to_rgb_aligned(depth, depth_intr, color_intr, T_l, color.shape)
    # print(aligned_depth.max())
    aligned_depth[aligned_depth>5000] = 0
    # print(aligned_depth.max())

    depth_normalized = cv2.normalize(aligned_depth, None, 0, 255, cv2.NORM_MINMAX)
    # print(list(depth_normalized))
    depth_8bit = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    combined_image = cv2.vconcat([color, depth_colored])
    cv2.namedWindow('RGB and Depth', cv2.WINDOW_NORMAL)
    cv2.imshow('RGB and Depth', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
