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

serial_nums = [
    {0: '213622251272', 1: '213522250729', 2: '037522250789'},
    {0: '213622251272', 1: '037522250789', 2: '213522250729'},
    {0: '213522250729', 1: '213622251272', 2: '037522250789'},
    {0: '213522250729', 1: '037522250789', 2: '213622251272'},
    {0: '037522250789', 1: '213622251272', 2: '213522250729'},
    {0: '037522250789', 1: '213522250729', 2: '213622251272'}
]



import camera_utils as cam
import camera_refactor
import visualization_utils as vis
import calibration_utils as calib

def set_up_tools():
    cam.view_live_camera_streams()  
    calib.write_camera_intrinsics_to_file()
    calib.run_calibrations(serial_nums)
    t_matrices = calib.get_transformation_matrices(serial_nums)
    if t_matrices:
        vis.view_combined_pcd(serial_nums, t_matrices)
    return t_matrices

if __name__ == "__main__":
    t_matrices = set_up_tools()
    if t_matrices:
        vis.view_combined_pcd(serial_nums, t_matrices)
    serial_list = list(serial_nums[0].values())
    camera = camera_refactor.CameraRecorder()
    print("Recording from cameras:", serial_list)

    camera.start_recording()
        # cam.record_audio_video(serial_list)