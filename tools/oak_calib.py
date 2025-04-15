#!/usr/bin/env python3

import depthai as dai
import numpy as np

RGB_RESOLUTION = (1920, 1080)
DEPTH_RESOLUTION = (1280, 720)

# Connect Device
with dai.Device() as device:
    calibData = device.readCalibration()

    M_rgb, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
    print(f"\nRGB Camera Default intrinsics...")
    print(M_rgb)
    print(width)
    print(height)

    if "OAK-1" in calibData.getEepromData().boardName or "BW1093OAK" in calibData.getEepromData().boardName:
        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720))
        print("RGB Camera resized intrinsics...")
        print(M_rgb)

        D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
        print("RGB Distortion Coefficients...")
        [print(name + ": " + value) for (name, value) in
         zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
             [str(data) for data in D_rgb])]

        print(f'RGB FOV {calibData.getFov(dai.CameraBoardSocket.CAM_A)}')

    else:
        M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, *RGB_RESOLUTION))
        print(f"\nRGB Camera resized intrinsics... {RGB_RESOLUTION[0]}, {RGB_RESOLUTION[1]} ")
        print(M_rgb)


        M_left, width, height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_B)
        print("\nLEFT Camera Default intrinsics...")
        print(M_left)
        print(width)
        print(height)

        M_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, *DEPTH_RESOLUTION))
        print(f"\nLEFT Camera resized intrinsics...  {DEPTH_RESOLUTION[0]}, {DEPTH_RESOLUTION[1]}")
        print(M_left, "\n")

        D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
        print("\nLEFT Distortion Coefficients...")
        [print(name+": "+value) for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4","τx","τy"],[str(data) for data in D_left])]

        D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
        print("\nRIGHT Distortion Coefficients...")
        [print(name+": "+value) for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4","τx","τy"],[str(data) for data in D_right])]

        print(f"\nRGB FOV {calibData.getFov(dai.CameraBoardSocket.CAM_A)}, Mono FOV {calibData.getFov(dai.CameraBoardSocket.CAM_B)}")

        R1 = np.array(calibData.getStereoLeftRectificationRotation())
        R2 = np.array(calibData.getStereoRightRectificationRotation())
        M_right = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), 1280, 720))

        H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
        print("\nLEFT Camera stereo rectification matrix...")
        print(H_left)

        H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))
        print("\nRIGHT Camera stereo rectification matrix...")
        print(H_right)

        """
        Transformation matrix composition: ??, [cm?]
            T = [
            [r00, r01, r02, tx],
            [r10, r11, r12, ty],
            [r20, r21, r22, tz],
            [0,   0,   0,   1]
            ]
            
            Point transformation:
            point_src = np.array([x, y, z, 1])
            point_dst = T @ point_src
        """

        lr_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C))
        print("\nTransformation matrix of where left Camera is W.R.T right Camera's optical center")
        print(lr_extrinsics)

        l_rgb_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A))
        print("\nTransformation matrix of where left Camera is W.R.T RGB Camera's optical center")
        print(l_rgb_extrinsics)

        r_rgb_extrinsics = np.array(
            calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A))
        print("\nTransformation matrix of where right Camera is W.R.T RGB Camera's optical center")
        print(r_rgb_extrinsics)