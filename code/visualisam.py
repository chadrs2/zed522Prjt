"""
GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)
See LICENSE for the license information
A visualSLAM example for the structure-from-motion problem on a simulated dataset
This version uses iSAM to solve the problem incrementally
"""

from __future__ import print_function

import sys
import math
import numpy as np
import cv2
import pyzed.sl as sl
import gtsam
from gtsam.examples import SFMdata
from gtsam import (Cal3_S2, GenericProjectionFactorCal3_S2,
                   NonlinearFactorGraph, NonlinearISAM, Point2, Pose3,
                   PriorFactorPoint3, PriorFactorPose3, Rot3,
                   PinholeCameraCal3_S2, Values, Point3)
from gtsam.symbol_shorthand import X, L

def main():
    """
    A structure-from-motion example with landmarks
    - The landmarks form a 10 meter cube
    - The robot rotates around the landmarks, always facing towards the cube
    """

    # Define the camera calibration parameters
    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()

    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    calibration_params = zed.get_camera_information().calibration_parameters
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy
    K = Cal3_S2(fx, fy, 0.0, cx, cy)

    # Define the camera observation noise model
    camera_noise = gtsam.noiseModel.Isotropic.Sigma(
        2, 1.0)  # one pixel in u and v

    # # Create the set of ground-truth landmarks
    # points = SFMdata.createPoints()
    # # Create the set of ground-truth poses
    # poses = SFMdata.createPoses(K)

    # Create a NonlinearISAM object which will relinearize and reorder the variables
    # every "reorderInterval" updates
    isam = NonlinearISAM(reorderInterval=3)

    # Create a Factor Graph and Values to hold the new data
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    # Loop over the different poses, adding the observations to iSAM incrementally
    #-- SIFT OBJECT
    # Setting Hessian Threshold and Creating SIFT object
    hessian_thresh = 50
    sift = cv2.xfeatures2d.SIFT_create(hessian_thresh)

    runtime = sl.RuntimeParameters()
    left_cam_rgba = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    depth_map = sl.Mat()
    point_cloud = sl.Mat()

    prev_img = np.empty((zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height,3),dtype=np.uint8)
    prev_kp = None
    prev_des = None
    prev_kp_dict = {}
    total_obsv_features = 0
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Get frame count
            i = zed.get_svo_position()

            # A new image and depth is available if grab() returns SUCCESS
            zed.retrieve_image(left_cam_rgba, sl.VIEW.LEFT) # Retrieve left image
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # Retrieve point cloud
            # Get left_cam image
            img2_rgba = left_cam_rgba.get_data()
            img2_rgb = cv2.cvtColor(img2_rgba, cv2.COLOR_RGBA2RGB)
            img2_gray = cv2.cvtColor(img2_rgba, cv2.COLOR_RGBA2GRAY)

            # Extract features from image
            key_pts2, descriptors2 = sift.detectAndCompute(img2_gray,None)

            # # Add factors for each landmark observation
            # for j, point in enumerate(key_pts2):
            #     pix_pt = list(int(k) for k in point.pt)
            #     measurement = Point2(pix_pt[0], pix_pt[1])
            #     factor = GenericProjectionFactorCal3_S2(
            #         measurement, camera_noise, X(i), L(j), K)
            #     graph.push_back(factor)

            # Initialize camera frame variables (TODO: poss. change these initialization params)
            noise = Pose3(r=Rot3.Rodrigues(-0.1, 0.2, 0.25),
                          t=Point3(0.05, -0.10, 0.20))
            initial_xi = noise

            # Add an initial guess for the current pose
            initial_estimate.insert(X(i), initial_xi)
        
            # If this is the first iteration, add a prior on the first pose to set the coordinate frame
            # and a prior on the first landmark to set the scale
            # Also, as iSAM solves incrementally, we must wait until each is observed at least twice before
            # adding it to iSAM.
            if i == 0:
                # Add factors for each landmark observation
                j = 0
                for point in key_pts2:
                    pix_pt = list(int(k) for k in point.pt)
                    err, point3D = point_cloud.get_value(pix_pt[0],pix_pt[1])
                    if not (math.isnan(point3D[0]) or math.isnan(point3D[1]) or math.isnan(point3D[2])):
                        measurement = Point2(pix_pt[0], pix_pt[1])
                        factor = GenericProjectionFactorCal3_S2(
                            measurement, camera_noise, X(i), L(j), K)
                        graph.push_back(factor)
                        j += 1

                # Add a prior on pose x0, with 0.3 rad std on roll,pitch,yaw and 0.1m x,y,z
                pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
                pose0 = Pose3(
                    Rot3(1,0,0, 0,1,0, 0,0,1),
                    Point3(0,0,0)
                )
                factor = PriorFactorPose3(X(0), pose0, pose_noise)
                graph.push_back(factor)

                # Add a prior on landmark l0
                point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
                pt0 = Point3(0,0,0)
                factor = PriorFactorPoint3(L(0), pt0, point_noise)
                graph.push_back(factor)

                # Add initial guesses to all observed landmarks
                j = 0
                for point in key_pts2:
                    # Get the 3D point cloud values for pixel (i,j)
                    pix_pt = list(int(k) for k in point.pt)
                    err, point3D = point_cloud.get_value(pix_pt[0],pix_pt[1])
                    init_lj = Point3(point3D[0],point3D[1],point3D[2])
                    if not (math.isnan(point3D[0]) or math.isnan(point3D[1]) or math.isnan(point3D[2])):
                        initial_estimate.insert(L(j), init_lj)
                        print(init_lj)
                        total_obsv_features += 1
                        # Add to dictionary
                        prev_kp_dict[j] = j
                        j += 1
                # print(prev_kp_dict)

                # Update previous variables
                prev_kp = key_pts2
                prev_des = descriptors2
                prev_img = img2_rgb
            else:
                curr_kp_dict = {}
                #-- Matching descriptor vectors with a FLANN based matcher
                # Since SIFT is a floating-point descriptor NORM_L2 is used
                matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
                knn_matches = matcher.knnMatch(prev_des, descriptors2, 2)
                #-- Filter matches using the Lowe's ratio test
                ratio_thresh = 0.7
                good_matches = []
                for m,n in knn_matches:
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                
                # Appropriately add in factors correlated to matched features from previous image        
                for match in good_matches:
                    prev_img_feat_idx = match.queryIdx
                    curr_img_feat_idx = match.trainIdx
                    if prev_img_feat_idx in prev_kp_dict:
                        j = prev_kp_dict[prev_img_feat_idx]

                        point = key_pts2[curr_img_feat_idx]
                        pix_pt = list(int(k) for k in point.pt)
                        err, point3D = point_cloud.get_value(pix_pt[0],pix_pt[1])
                        if not (math.isnan(point3D[0]) or math.isnan(point3D[1]) or math.isnan(point3D[2])):
                            measurement = Point2(pix_pt[0],pix_pt[1])
                            factor = GenericProjectionFactorCal3_S2(
                                measurement, camera_noise, X(i), L(j), K)
                            graph.push_back(factor)
                            curr_kp_dict[curr_img_feat_idx] = j

                # print(curr_kp_dict)
                # print("Total observed features:",total_obsv_features)
                # Add in remaining factors that have been newly observed
                for l, point in enumerate(key_pts2):
                    # print("L",l,"--",curr_kp_dict)
                    if not (l in curr_kp_dict):
                        # print("Poss inserting new landmark")
                        pix_pt = list(int(k) for k in point.pt)
                        err, point3D = point_cloud.get_value(pix_pt[0],pix_pt[1])
                        if not (math.isnan(point3D[0]) or math.isnan(point3D[1]) or math.isnan(point3D[2])):
                            # print("Inserting new landmark")
                            curr_kp_dict[l] = total_obsv_features # value is last index
                            measurement = Point2(pix_pt[0],pix_pt[1])
                            factor = GenericProjectionFactorCal3_S2(
                                measurement, camera_noise, X(i), L(curr_kp_dict[l]), K)
                            graph.push_back(factor)

                            # Add initial guesses to newly observed landmarks
                            init_lj = Point3(point3D[0],point3D[1],point3D[2])
                            # print(init_lj)
                            initial_estimate.insert(L(curr_kp_dict[l]), init_lj)
                            total_obsv_features += 1
                # print(curr_kp_dict)

                # Update iSAM with the new factors
                isam.update(graph, initial_estimate)
                current_estimate = isam.estimate()
                # print('*' * 50)
                # print('Frame {}:'.format(i))
                # current_estimate.print_('Current estimate: ')

                # Clear the factor graph and values for the next iteration
                graph.resize(0)
                initial_estimate.clear()

                # Update previous variables
                prev_kp_dict = curr_kp_dict
                prev_kp = key_pts2
                prev_des = descriptors2
                prev_img = img2_rgb

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO end has been reached.")# Looping back to first frame")
            # zed.set_svo_position(0)
            break

    zed.close()
    print("\nFINISH")

if __name__ == '__main__':
    main()