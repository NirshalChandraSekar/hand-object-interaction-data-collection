import pyrealsense2 as rs

# reset the hardware
def reset_hardware():
    ctx = rs.context()
    devices = ctx.query_devices()
    for device in devices:
        print(f"Resetting device: {device.get_info(rs.camera_info.name)} (Serial: {device.get_info(rs.camera_info.serial_number)})")
        device.hardware_reset()
    print("All devices have been reset.")

    print("Hardware reset complete.")

# Example usage
if __name__ == "__main__":
    reset_hardware()
    print("Hardware reset function executed successfully.")
