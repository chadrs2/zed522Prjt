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
import matplotlib.pyplot as plt
import cv2
import pyzed.sl as sl
import gtsam
from gtsam.examples import SFMdata
from gtsam import (Cal3_S2, GenericProjectionFactorCal3_S2,
                   NonlinearFactorGraph, NonlinearISAM, Point2, Pose3,
                   PriorFactorPoint3, PriorFactorPose3, Rot3,
                   PinholeCameraCal3_S2, Values, Point3)
from gtsam.symbol_shorthand import X, L
import gtsam.utils.plot as gtsam_plot
from mpl_toolkits.mplot3d import Axes3D


def visual_ISAM2_plot(result):
    """
    VisualISAMPlot plots current state of ISAM2 object
    Author: Ellon Paiva
    Based on MATLAB version by: Duy Nguyen Ta and Frank Dellaert
    """

    # Declare an id for the figure
    fignum = 0

    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plt.cla()

    # Plot points
    # Can't use data because current frame might not see all points
    # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
    # gtsam.plot_3d_points(result, [], marginals)
    gtsam_plot.plot_3d_points(fignum, result, 'rx')

    # Plot cameras
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 10)
        i += 1

    # draw
    axes.set_xlim3d(-40, 40)
    axes.set_ylim3d(-40, 40)
    axes.set_zlim3d(-40, 40)
    plt.pause(1)

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

    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Get frame count
            i = zed.get_svo_position()
            print("------ i = ", i)
            if i >= 4:
                # only get first 2 frames
                print("finished retrieving two images")
                break

            # A new image and depth is available if grab() returns SUCCESS
            zed.retrieve_image(left_cam_rgba, sl.VIEW.LEFT) # Retrieve left image
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # Retrieve point cloud
            
            if i == 0:
                # Get left_cam image
                img1_rgba = left_cam_rgba.get_data()
                img1 = cv2.cvtColor(img1_rgba, cv2.COLOR_RGBA2RGB)
                img1_gray = cv2.cvtColor(img1_rgba, cv2.COLOR_RGBA2GRAY)

                # Extract features from image
                key_pts1, descriptors1 = sift.detectAndCompute(img1_gray,None)
            elif i == 1:
                # Get left_cam image
                img2_rgba = left_cam_rgba.get_data()
                img2 = cv2.cvtColor(img2_rgba, cv2.COLOR_RGBA2RGB)
                img2_gray = cv2.cvtColor(img2_rgba, cv2.COLOR_RGBA2GRAY)

                # Extract features from image
                key_pts2, descriptors2 = sift.detectAndCompute(img2_gray,None)

    #-- Matching descriptor vectors with a FLANN based matcher
    # Since SIFT is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    
    # Initialize camera frame variables (TODO: poss. change these initialization params)
    initial_xi = Pose3(
        Rot3(1,0,0, 0,1,0, 0,0,1),
        Point3(0,0,0)
    )
    # Add an initial guess for the current pose
    initial_estimate.insert(X(0), initial_xi)
    initial_estimate.insert(X(1), initial_xi)

    # Add a prior on pose x0, with 0.3 rad std on roll,pitch,yaw and 0.1m x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.3, 0.3, 0.3, 0.1*1000, 0.1*1000, 0.1*1000]))
    pose0 = Pose3(
        Rot3(1,0,0, 0,1,0, 0,0,1),
        Point3(0,0,0)
    )
    factor = PriorFactorPose3(X(0), pose0, pose_noise)
    graph.push_back(factor)

    curr_kp_dict = []
    total_obsv_features = 0
    j = 0
    for match in good_matches:
        img1_feat_idx = match.queryIdx
        img2_feat_idx = match.trainIdx

        pt1 = key_pts1[img1_feat_idx]
        pix_pt = list(int(k) for k in pt1.pt)
        err, point3D = point_cloud.get_value(pix_pt[0],pix_pt[1])
        if not (math.isnan(point3D[0]) or math.isnan(point3D[1]) or math.isnan(point3D[2])):
            measurement = Point2(pix_pt[0],pix_pt[1])
            factor = GenericProjectionFactorCal3_S2(
                measurement, camera_noise, X(0), L(j), K)
            graph.push_back(factor)

            curr_kp_dict.append(img1_feat_idx) # newly added

            init_lj = Point3(point3D[0],point3D[1],point3D[2])
            initial_estimate.insert(L(j), init_lj)
            total_obsv_features += 1

            pt2 = key_pts2[img2_feat_idx]
            pix_pt2 = list(int(k) for k in pt2.pt)
            measurement = Point2(pix_pt2[0],pix_pt2[1])
            factor = GenericProjectionFactorCal3_S2(
                measurement, camera_noise, X(1), L(j), K)
            graph.push_back(factor)

            if True:
                point_noise = gtsam.noiseModel.Isotropic.Sigma(3, .1*1000)
                pt0 = Point3(point3D[0],point3D[1],point3D[2])
                # pt0 = Point3(0,0,0)
                factor = PriorFactorPoint3(L(j), pt0, point_noise)
                graph.push_back(factor)
            j += 1
    # print(total_obsv_features)
    for l, point in enumerate(key_pts1):
        # print("L",l,"--",curr_kp_dict)
        if not (l in curr_kp_dict):
            # print("Poss inserting new landmark")
            pix_pt = list(int(k) for k in point.pt)
            err, point3D = point_cloud.get_value(pix_pt[0],pix_pt[1])
            if not (math.isnan(point3D[0]) or math.isnan(point3D[1]) or math.isnan(point3D[2])):
                # print("Inserting new landmark")
                # curr_kp_dict[l] = total_obsv_features # value is last index
                measurement = Point2(pix_pt[0],pix_pt[1])
                factor = GenericProjectionFactorCal3_S2(
                    measurement, camera_noise, X(0), L(total_obsv_features), K)
                # print("total observations: ", total_obsv_features)
                graph.push_back(factor)

                point_noise = gtsam.noiseModel.Isotropic.Sigma(3, .1*1000)
                pt0 = Point3(point3D[0],point3D[1],point3D[2])
                # pt0 = Point3(0,0,0)
                factor = PriorFactorPoint3(L(total_obsv_features), pt0, point_noise)
                graph.push_back(factor)

                init_lj = Point3(point3D[0],point3D[1],point3D[2])
                initial_estimate.insert(L(total_obsv_features), init_lj)
                total_obsv_features += 1

    # Update iSAM with the new factors
    isam.update(graph, initial_estimate)
    current_estimate = isam.estimate()
    print('*' * 50)
    print('Frame {}:'.format(i))
    current_estimate.print_('Current estimate: ')
    
    # plotEstimates()
    # plt.ion()
    # visual_ISAM2_plot(current_estimate)
    # plt.ioff()
    # Declare an id for the figure
    fignum = 0

    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plt.cla()

    # Plot points
    # Can't use data because current frame might not see all points
    # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
    # gtsam.plot_3d_points(result, [], marginals)
    gtsam_plot.plot_3d_points(fignum, current_estimate, 'rx')

    # Plot cameras
    i = 0
    while current_estimate.exists(X(i)):
        pose_i = current_estimate.atPose3(X(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 10)
        i += 1

    axes.set_xlim3d(-40, 40)
    axes.set_ylim3d(-40, 40)
    axes.set_zlim3d(-40, 40)

    plt.show()

    # Clear the factor graph and values for the next iteration
    graph.resize(0)
    initial_estimate.clear()

    zed.close()
    print("\nFINISH")

if __name__ == '__main__':
    main()