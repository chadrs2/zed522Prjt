#!usr/bin/python3

import sys
import pyzed.sl as sl
import numpy as np
import cv2

def main():
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
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Get frame count
            svo_position = zed.get_svo_position()

            # A new image and depth is available if grab() returns SUCCESS
            zed.retrieve_image(left_cam_rgba, sl.VIEW.LEFT) # Retrieve left image
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) # Retrieve point cloud

            # Get left_cam image
            img2_rgba = left_cam_rgba.get_data()
            img2_rgb = cv2.cvtColor(img2_rgba, cv2.COLOR_RGBA2RGB)
            img2_gray = cv2.cvtColor(img2_rgba, cv2.COLOR_RGBA2GRAY)
            # cv2.imshow("Image",img1_rgb)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Extract features from image
            key_pts2, descriptors2 = sift.detectAndCompute(img2_gray,None)
            # img1_kp = np.empty((img1_rgb.shape[0],img1_rgb.shape[1],img1_rgb.shape[2]),dtype=np.uint8)
            # cv2.drawKeypoints(img1_rgb,key_pts1,img1_kp)
            # cv2.imshow("Keypoint Image",img1_kp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if prev_kp == None:
                prev_kp = key_pts2
                prev_des = descriptors2
                prev_img = img2_rgb
            else:
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
                #-- Draw matches
                img_matches = np.empty((max(prev_img.shape[0], img2_rgb.shape[0]), prev_img.shape[1]+img2_rgb.shape[1], 3), dtype=np.uint8)
                cv2.drawMatches(prev_img, prev_kp, img2_rgb, key_pts2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                #-- Show detected matches
                cv2.imshow('Good Matches', img_matches)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Update previous variables
                prev_kp = key_pts2
                prev_des = descriptors2
                prev_img = img2_rgb
            
            # Get the 3D point cloud values for pixel (i,j)
            #point3D = point_cloud.get_value(i,j)
            #x = point3D[0]; y = point3D[1]; z = point3D[2]

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO end has been reached.")# Looping back to first frame")
            # zed.set_svo_position(0)
            break

    zed.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()