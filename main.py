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

serial_nums = {0: '213522250729', 1:'213622251272', 2: '037522250789'}

import camera_utils as cam
import visualization_utils as vis
import calibration_utils as calib

if __name__ == "__main__":
    cam.view_live_camera_streams()  
    calib.write_camera_intrinsics_to_file()
    # '213522250729','037522250789'
    # 213522250729, '213622251272'
    # '213622251272','037522250789' both
    t_matrices = calib.run_calibrations(serial_nums)
    cam.save_images()
    vis.view_combined_pcd(serial_nums, t_matrices)
    print("Calibration completed and point cloud visualized.")

    cam.record_audio_video(t_matrices)