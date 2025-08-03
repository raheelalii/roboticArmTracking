import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import os
import ctypes
from ctypes import c_float, c_int

# Try to load the C++ robot control DLL
robot_dll = None
robot_initialized = False

try:
    # Check if DLL exists
    dll_path = './robot_control.dll'
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL not found at {dll_path}")
    
    print(f"Loading C++ DLL from: {os.path.abspath(dll_path)}")
    
    # Load the C++ DLL
    robot_dll = ctypes.CDLL(dll_path)
    
    # Define function signatures
    robot_dll.initialize_robot.restype = c_int
    robot_dll.move_robot_to_offset.argtypes = [c_float, c_float, c_float]
    robot_dll.move_robot_to_offset.restype = c_int
    robot_dll.cleanup_robot.restype = None
    
    print("✓ C++ Robot control DLL loaded successfully")
    
    # Initialize robot connection
    print("Initializing robot connection...")
    result = robot_dll.initialize_robot()
    if result == 0:
        robot_initialized = True
        print("✓ Robot initialized successfully via C++")
    else:
        print(f"✗ Robot initialization failed with code: {result}")
        print("Note: Make sure robot is connected and in automatic mode")
        
except FileNotFoundError as e:
    print(f"✗ DLL file not found: {e}")
    print("Please run 'build_test_move.bat' to build the robot_control.dll")
except Exception as e:
    print(f"✗ Failed to load C++ robot control: {e}")
    print("Running in camera-only mode")

print(f"Robot DLL loaded: {robot_dll is not None}")
print(f"Robot initialized: {robot_initialized}")

# Global variables for depth frame and intrinsics (updated in the main loop)
depth_frame_global = None
intrinsics_global = None

def move_robot_to_point(x_mm, y_mm, z_mm):
    """Move robot using C++ DLL function"""
    if not robot_initialized or robot_dll is None:
        print("Robot control not available - C++ DLL not loaded or robot not initialized")
        return
    
    try:
        print(f"Calling C++ function to move robot by offset: ({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f}) mm")
        
        # Call the C++ function to move robot by offset
        result = robot_dll.move_robot_to_offset(c_float(x_mm), c_float(y_mm), c_float(z_mm))
        
        if result == 0:
            print("✓ Robot movement completed successfully via C++")
        elif result == -1:
            print("✗ Robot movement failed - robot not initialized or error occurred")
        elif result == -2:
            print("✗ Robot movement timeout")
        elif result == 101:
            print("✗ Robot movement failed - Error 101 (possibly joint limits or safety constraints)")
        elif result == 112:
            print("✗ Robot movement failed - Error 112 (possibly position out of reach)")
        else:
            print(f"✗ Robot movement failed with error code: {result}")
        
    except Exception as e:
        print(f"Error calling C++ robot function: {e}")

def mouse_callback(event, px, py, flags, param):
    global depth_frame_global, intrinsics_global
    if depth_frame_global is None or intrinsics_global is None:
        return

    # Get depth value at the pixel (in meters)
    depth = depth_frame_global.get_distance(px, py)

    if depth == 0:
        print("No depth data available at this point.")
        return

    # Deproject pixel to 3D point (in meters)
    depth_point = rs.rs2_deproject_pixel_to_point(intrinsics_global, [px, py], depth)

    # Convert to millimeters
    x_mm = depth_point[0] * 1000
    y_mm = depth_point[1] * 1000
    z_mm = depth_point[2] * 1000

    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Hover: X: {x_mm:.2f} mm, Y: {y_mm:.2f} mm, Z: {z_mm:.2f} mm")

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Click: Camera coordinates - X: {x_mm:.2f} mm, Y: {y_mm:.2f} mm, Z: {z_mm:.2f} mm")
        
        # Map camera coordinates to robot TCP coordinates:
        # Camera X → Robot X (left/right movement)
        # Camera Y → Robot Y (up/down movement)  
        # Camera Z → Robot Z (forward/backward movement)
        robot_x = x_mm  # Camera X becomes Robot X
        robot_y = y_mm  # Camera Y becomes Robot Y
        robot_z = z_mm  # Camera Z becomes Robot Z
        
        print(f"Robot coordinates - X: {robot_x:.2f} mm, Y: {robot_y:.2f} mm, Z: {robot_z:.2f} mm")
        
        # Move robot to the mapped coordinates
        move_robot_to_point(robot_x, robot_y, robot_z)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Create OpenCV window
cv2.namedWindow('Live Feed')
cv2.setMouseCallback('Live Feed', mouse_callback)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned color and depth frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Update globals for mouse callback
        depth_frame_global = depth_frame
        intrinsics_global = color_frame.profile.as_video_stream_profile().intrinsics

        # Convert color frame to numpy array for display
        color_image = np.asanyarray(color_frame.get_data())

        # Display the live feed
        cv2.imshow('Live Feed', color_image)

        # Exit on 'q' key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Cleanup robot connection
    if robot_initialized and robot_dll is not None:
        try:
            robot_dll.cleanup_robot()
            print("✓ Robot cleanup completed")
        except Exception as e:
            print(f"Robot cleanup error: {e}")