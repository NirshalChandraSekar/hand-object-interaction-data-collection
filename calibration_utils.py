import pyrealsense2 as rs
import subprocess
import re
import numpy as np
import os
from PIL import Image
import h5py
import cv2
import time

def write_camera_intrinsics_to_file():
    """
    Connects to all RealSense cameras connected to the system,
    retrieves the color camera intrinsics, and writes them to XML files
    in a 'calibration/' folder named by their serial numbers.

    Each XML file contains camera parameters formatted for use with
    Calibu or similar calibration tools.

    If no devices are found, prints a message and returns.
    """
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    for i, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)

        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        started = False
        try:
            pipeline_profile = pipeline.start(config)
            started = True

            color_stream = pipeline_profile.get_stream(rs.stream.color)
            video_profile = color_stream.as_video_stream_profile()
            intrinsics = video_profile.get_intrinsics()

            with open(f'calibration/{serial}.xml', "w") as f:
                     f.write(f"""
<rig>
    <camera>
        <camera_model name="" index="0" serialno="{serial}" type="calibu_fu_fv_u0_v0_k1_k2_k3" version="0">
            <width> {intrinsics.width} </width>
            <height> {intrinsics.height} </height>
            <!-- Use RDF matrix, [right down forward], to define the coordinate frame convention -->
            <right> [ 1; 0; 0 ] </right>
            <down> [ 0; 1; 0 ] </down>
            <forward> [ 0; 0; 1 ] </forward>
            <!-- Camera parameters ordered as per type name. -->
            <params> [ {intrinsics.fx}; {intrinsics.fy}; {intrinsics.ppx}; {intrinsics.ppy}; 0.000; 0.000; 0.000 ] </params>
        </camera_model>
        <pose>
            <!-- Camera pose. World from Camera point transfer. 3x4 matrix, in the RDF frame convention defined above -->
            <T_wc> [ 1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0 ] </T_wc>
        </pose>
    </camera>
</rig>
""")
        except Exception as e:
            print(f"Failed to get color intrinsics from device {serial}: {e}")
        finally:
            if started:
                pipeline.stop()

def create_calibration_images_live(serial_numbers, num_frames=64):
   """
    Captures color frames live from multiple RealSense cameras simultaneously
    for the purpose of calibration. Saves captured images to 'calibration/calib_images'.

    Interactive controls:
        - Press 's' to save the current frames from all cameras.
        - Press 'q' to quit capturing before reaching num_frames.

    If OpenCV windows are not available (e.g., no display), auto-captures frames
    every 2 seconds without preview.

    Parameters:
    -----------
    serial_numbers : list of str
        List of serial numbers of the RealSense cameras to capture from.
    num_frames : int, optional
        Number of frame sets to capture (default is 64).
    """
   os.makedirs("calibration/calib_images", exist_ok=True)
  
   # Initialize pipelines for each camera
   pipelines = []
   configs = []
  
   try:
       # Setup pipelines for each camera
       for serial in serial_numbers:
           pipeline = rs.pipeline()
           config = rs.config()
           config.enable_device(serial)
           config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
          
           pipeline.start(config)
           pipelines.append(pipeline)
           configs.append(config)
           print(f"Started camera {serial}")
      
       print("Press 's' to save a frame, 'q' to quit after capturing enough frames")
       print("Move the calibration pattern to different positions and capture frames")
      
       frame_count = 0
      
       try:          
           while frame_count < num_frames:  # Capture up to 64 frames
               frames_data = []
              
               # Get frames from all cameras
               for i, pipeline in enumerate(pipelines):
                   frames = pipeline.wait_for_frames()
                   color_frame = frames.get_color_frame()
                  
                   if color_frame:
                       # Convert to numpy array
                       color_image = np.asanyarray(color_frame.get_data())
                       frames_data.append(color_image)
              
               # Display frames (optional - requires display)
               if len(frames_data) >= 2:
                   try:
                       # Show preview windows
                       cv2.imshow(f'Camera {serial_numbers[0]}', frames_data[0])
                       cv2.imshow(f'Camera {serial_numbers[1]}', frames_data[1])
                      
                       key = cv2.waitKey(1) & 0xFF
                      
                       if key == ord('s'):  # Save frame
                           # Convert BGR to RGB for PIL
                           for i, frame in enumerate(frames_data):
                               rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                               img = Image.fromarray(rgb_frame)
                               img.save(f"calibration/calib_images/cam{i}_{frame_count:04}.png")
                          
                           print(f"Saved frame set {frame_count}")
                           frame_count += 1
                          
                       elif key == ord('q'):  # Quit
                           break
                          
                   except cv2.error:
                       # If no display available, auto-capture frames with delay
                       print(f"Auto-capturing frame {frame_count} (no display available)")
                      
                       for i, frame in enumerate(frames_data):
                           rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                           img = Image.fromarray(rgb_frame)
                           img.save(f"calibration/calib_images/cam{i}_{frame_count:04}.png")
                      
                       frame_count += 1
                       import time
                       time.sleep(2)  # Wait 2 seconds between captures
                      
       except ImportError:
           print("OpenCV not available, capturing frames automatically...")
           # Fallback: capture frames automatically without preview
           for frame_count in range(64):
               frames_data = []
              
               for i, pipeline in enumerate(pipelines):
                   frames = pipeline.wait_for_frames()
                   color_frame = frames.get_color_frame()
                  
                   if color_frame:
                       color_image = np.asanyarray(color_frame.get_data())
                       # Convert BGR to RGB
                       rgb_image = color_image[..., ::-1]  # BGR to RGB
                       img = Image.fromarray(rgb_image)
                       img.save(f"calib_images/cam{i}_{frame_count:04}.png")
              
               print(f"Captured frame set {frame_count}")
               import time
               time.sleep(1)  # Wait 1 second between captures
  
   finally:
       # Clean up
       for pipeline in pipelines:
           pipeline.stop()
       try:
           cv2.destroyAllWindows()
       except:
           pass
      
       print(f"Captured {frame_count} frame sets for calibration")


def create_calibration_images(dataset_path, num_frames = 64):
   """
    Extracts color calibration images from an existing HDF5 dataset and saves them
    as PNG files in 'calibration/calib_images'.

    The function assumes the dataset contains multiple cameras with color frames
    organized under serial number groups.

    Parameters:
    -----------
    dataset_path : str
        File path to the HDF5 dataset.
    num_frames : int, optional
        Number of frames to extract from each camera (default is 64).
    """
   if not os.path.exists(dataset_path):
       print(f"Dataset path {dataset_path} does not exist.")
       return
   
   os.makedirs("calibration/calib_images", exist_ok=True)

   with h5py.File(dataset_path, "r") as data:
       serials = list(data.keys())
       print(serials)


       os.makedirs("calib_images", exist_ok=True)
       max_frames = len(data[f'{serials[0]}/frames/color'])


       for i in range(num_frames):
           color_image_0 = data[f'{serials[0]}/frames/color'][str(i % max_frames)][()]
           color_image_1 = data[f'{serials[1]}/frames/color'][str(i % max_frames)][()]
          
           # If the image is BGR, convert to RGB
           if color_image_0.shape[2] == 3:  # Assuming H x W x C
               color_image_0 = color_image_0[..., ::-1]  # BGR to RGB
               color_image_1 = color_image_1[..., ::-1]

           Image.fromarray(color_image_0).save(f"calibration/calib_images/cam0_{i:04}.png")
           Image.fromarray(color_image_1).save(f"calibration/calib_images/cam1_{i:04}.png")

def run_calibrations(serial_numbers, offline_calibration = False, dataset_path = None):
    """
    Runs camera calibration for adjacent camera pairs using either live capture
    or an offline dataset.

    For each adjacent pair, it updates a shell script with the correct camera IDs,
    executes the calibration script, then parses the resulting XML to extract
    the transformation matrix (pose) between the camera pair.

    Parameters:
    -----------
    serial_numbers : dict
        Dictionary mapping camera indices to serial numbers.
    offline_calibration : bool, optional
        If True, calibration images will be extracted from dataset_path; otherwise,
        live capture will be used (default is False).
    dataset_path : str or None, optional
        Path to offline dataset file; required if offline_calibration is True.

    Returns:
    --------
    dict
        Dictionary mapping camera pair strings (e.g., "ID0-ID1") to 4x4 numpy arrays
        representing the transformation matrix from camera 1 to camera 2.
    """
    script_path = "calibration/offline_calib.sh" if offline_calibration else "calibration/online_calib.sh"
    t_matrices = {}
    for i in range((len(serial_numbers) - 1)):
        id0 = f"{serial_numbers[i]}"
        id1 = f"{serial_numbers[i+1]}"
        print(f"Running calibration for {id0} and {id1}")

        if offline_calibration and dataset_path is None:
           print("Capturing live frames from cameras...")
           create_calibration_images_live([id0, id1])
        elif offline_calibration:
           print("Using offline dataset...")
           create_calibration_images(dataset_path)

        with open(script_path, "r") as f:
            content = f.read()

        # Replace values after export ID0=
        # Replace only the values after export ID0= and export ID1= without changing the rest
        content = re.sub(r"(export\s+ID0=)[^\s#]+", lambda m: m.group(1) + id0, content)
        content = re.sub(r"(export\s+ID1=)[^\s#]+", lambda m: m.group(1) + id1, content)


        with open(script_path, "w") as f:
            f.write(content)

        subprocess.run(["./" + script_path], check=True)

        output_path = f"calibration/{id0}-{id1}.xml"
        
        with open(output_path, "r") as f:
            content = f.read()

        matches = re.findall(r"<T_wc>\s*(.*?)\s*</T_wc>", content, re.DOTALL)
        t_matrix = np.eye(4)
        if matches:
            matrix_str = matches[-1].strip()

            # Clean and convert to NumPy array
            matrix_str = matrix_str.strip("[]")  # remove outer brackets
            rows = [r.strip() for r in matrix_str.split(";")]
            matrix = np.array([[float(num.strip()) for num in row.split(",")] for row in rows])
            #Correctly set the rotation and translation
            t_matrix[:3, :3] = matrix[:, :3]
            t_matrix[:3, 3] = matrix[:, 3]

        t_matrices[f"{id0}-{id1}"] = t_matrix

        time.sleep(3)

    print("Transformation matrices:")
    for key, value in t_matrices.items():
        print(f"{key}: {value}")
    return t_matrices