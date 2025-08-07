"""
Main script to handle camera calibration, visualization, image capture, and recording.
It uses utility modules for camera operations, visualization, and calibration tasks.

Steps performed:
1. View live streams from all connected cameras.
2. Retrieve intrinsic parameters for each camera.
3. Calibrate extrinsics between cameras.
4. Save captured images.
5. Visualize the combined point cloud from multiple cameras.
6. Record synchronized audio and video data.
"""

import camera_utils as cam
import visualization_utils as vis
import calibration_utils as calib

if "__name__" == "__main__":
    vis.view_live_camera_streams()
    serial_nums = calib.retrieve_intrinsics()
    t_matrices = calib.calibrate_extrinsics(serial_nums, True)
    cam.save_images()
    vis.view_combined_pcd(serial_nums, t_matrices)

    cam.record_audio_video()