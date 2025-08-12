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

# Dict order changes depending on calibration results
# Order is saved as the order of serial numbers in the recording file

serial_nums = {0: '037522250789', 1: '213522250729', 2:'213622251272'}

import camera_utils as cam
import visualization_utils as vis
import calibration_utils as calib

if __name__ == "__main__":
    cam.view_live_camera_streams()  
    calib.write_camera_intrinsics_to_file()
    calib.run_calibrations(serial_nums)
    t_matrices = calib.get_transformation_matrices(serial_nums)
    if t_matrices is not None:
        cam.save_images()
        vis.view_combined_pcd(serial_nums, t_matrices)
    serial_list = list(serial_nums.values())
    print("Recording from cameras:", serial_list)
    cam.record_audio_video(serial_list, t_matrices)