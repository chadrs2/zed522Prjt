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

    runtime = sl.RuntimeParameters()
    left_cam_rgba = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    depth_map = sl.Mat()
    point_cloud = sl.Mat()

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
            img1 = left_cam_rgba.get_data()
            cv2.imshow("Image",img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Get the 3D point cloud values for pixel (i,j)
            #point3D = point_cloud.get_value(i,j)
            #x = point3D[0]; y = point3D[1]; z = point3D[2]

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("SVO end has been reached. Looping back to first frame")
            zed.set_svo_position(0)
            break

    zed.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()