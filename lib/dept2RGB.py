import sys

import numpy as np
import cv2

def depth_to_rgb_aligned(depth_img, depth_intr, rgb_intr, T, rgb_shape):
    """
    Original code:
       for v in range(0, height_d, 1):
           for u in range(0, width_d, 1):
               z = depth_img[v, u] / 1000.0  # convert to m
               if z == 0:
                   continue

               # Deprojection the depth pixel to 3D poit in depth frame
               x = (u - cx_d) * z / fx_d
               y = (v - cy_d) * z / fy_d
               point_d = np.array([x, y, z, 1])

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
       """

    height_d, width_d = depth_img.shape
    height_rgb, width_rgb, __ = rgb_shape
    fx_d, fy_d = depth_intr[0][0], depth_intr[1][1]
    cx_d, cy_d = depth_intr[0][2], depth_intr[1][2]

    fx_rgb, fy_rgb = rgb_intr[0][0], rgb_intr[1][1]
    cx_rgb, cy_rgb = rgb_intr[0][2], rgb_intr[1][2]

    aligned_depth = np.zeros((height_rgb, width_rgb), dtype=np.uint16)

    # Creating coordinate grids for depth image
    u_d, v_d = np.meshgrid(np.arange(width_d), np.arange(height_d))

    # Transfer depth data to meters
    z = depth_img[v_d, u_d] / 1000.0

    # Mask for non zero depth
    valid_depth_mask = z > 0

    # Calculation of 3D points in the depth camera coordinate system for valid depths
    x_d = (u_d[valid_depth_mask] - cx_d) * z[valid_depth_mask] / fx_d
    y_d = (v_d[valid_depth_mask] - cy_d) * z[valid_depth_mask] / fy_d
    z_valid = z[valid_depth_mask]
    ones = np.ones_like(z_valid)
    points_d = np.stack([x_d, y_d, z_valid, ones], axis=-1)

    # Transforming 3D points into the RGB camera coordinate system
    points_rgb = points_d @ T.T  # We use T.T for correct multiplication of matrices with vectors as rows

    x_rgb = points_rgb[:, 0]
    y_rgb = points_rgb[:, 1]
    z_rgb = points_rgb[:, 2]

    # Masking points behind an RGB camera
    valid_rgb_z_mask = z_rgb > 0

    x_rgb_valid = x_rgb[valid_rgb_z_mask]
    y_rgb_valid = y_rgb[valid_rgb_z_mask]
    z_rgb_valid = z_rgb[valid_rgb_z_mask]

    # Projection of 3D points into the image plane of an RGB camera
    u_rgb = np.round((x_rgb_valid * fx_rgb) / z_rgb_valid + cx_rgb).astype(int)
    v_rgb = np.round((y_rgb_valid * fy_rgb) / z_rgb_valid + cy_rgb).astype(int)
    z_mm = np.round(z_rgb_valid * 1000).astype(int)

    # Masking points outside the RGB range of an image
    valid_rgb_coords_mask = (u_rgb >= 0) & (u_rgb < width_rgb) & (v_rgb >= 0) & (v_rgb < height_rgb)

    u_rgb_valid_in_frame = u_rgb[valid_rgb_coords_mask]
    v_rgb_valid_in_frame = v_rgb[valid_rgb_coords_mask]
    z_mm_valid_in_frame = z_mm[valid_rgb_coords_mask]

    # Updating aligned_depth using NumPy indexing
    aligned_depth_flat_indices = np.ravel_multi_index((v_rgb_valid_in_frame, u_rgb_valid_in_frame), aligned_depth.shape)

    # Creating an array for new depths and initializing to zero
    new_depths = np.zeros(aligned_depth.size, dtype=aligned_depth.dtype)
    np.put(new_depths, aligned_depth_flat_indices, z_mm_valid_in_frame)
    new_depths_reshaped = new_depths.reshape(aligned_depth.shape)

    # Update aligned_depth only where the new depth is smaller or the original is 0
    update_mask = (new_depths_reshaped < aligned_depth) | (aligned_depth == 0)
    aligned_depth[update_mask] = new_depths_reshaped[update_mask]

    return aligned_depth


if __name__ == "__main__":
    # Got for oak-d-pro-poe 192.168.1.53, left to rgb camera.
    fc = 1551.5
    c_xc = 966.4
    c_yc = 542.9

    color_intr = [[fc, 0, c_xc],
                  [0, fc, c_yc],
                  [0, 0, 1]]

    # right mono
    fd = 799.9
    c_xd = 635.5
    c_yd = 389.6

    depth_intr_r = [[fd, 0, c_xd],
                  [0, fd, c_yd],
                  [0, 0, 1]]
    # left mono
    fd = 796.8
    c_xd = 644.3
    c_yd = 360.8

    depth_intr_l = [[fd, 0, c_xd],
                    [0, fd, c_yd],
                    [0, 0, 1]]

    T_l = np.array([  # original displacement in cm, to be im meters
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

    data = np.load(sys.argv[1])
    depth = data["depth"]
    print(depth.shape)
    color = data["img"]
    print(color.shape)
    aligned_depth = depth_to_rgb_aligned(depth, depth_intr_r, color_intr, T_r, color.shape)
    # aligned_depth = depth_to_rgb_aligned(depth, depth_intr_l, color_intr, T_r, color.shape)
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
