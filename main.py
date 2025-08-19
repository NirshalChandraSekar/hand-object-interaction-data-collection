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

# Dict order changes depending on calibration results
# Order is saved as the order of serial numbers in the recording file

serial_nums = [
    {0: '213622251272', 1: '213522250729', 2: '037522250789'},
    {0: '213622251272', 1: '037522250789', 2: '213522250729'},
    {0: '213522250729', 1: '213622251272', 2: '037522250789'},
    {0: '213522250729', 1: '037522250789', 2: '213622251272'},
    {0: '037522250789', 1: '213622251272', 2: '213522250729'},
    {0: '037522250789', 1: '213522250729', 2: '213622251272'}
]

serial_dict = serial_nums[1]  # Default to first configuration

def set_up_tools():
    cam.view_live_camera_streams()  
    calib.write_camera_intrinsics_to_file()
    calib.run_calibrations(serial_dict)
   
if __name__ == "__main__":
    # set_up_tools()
    t_matrices = calib.get_transformation_matrices(serial_dict)
    if t_matrices:
        cam.save_images()
        vis.view_combined_pcd(serial_dict, t_matrices)
    serial_list = list(serial_dict.values())
    print("Recording from cameras:", serial_list)

    camera = cam.CameraRecorder()
    camera.start_recording(serial_list, t_matrices)
