import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import os
import ctypes
from ctypes import c_float, c_int
from ultralytics import YOLO
import threading
import time
from datetime import datetime

# Configuration variables
TARGET_LABEL = "bottle"  # Object to track (e.g., "bottle", "person", "cup")
ENABLE_TRACKING = True   # Enable/disable real-time tracking
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for detection
TRACKING_INTERVAL = 0.5  # Seconds between tracking updates
MIN_DEPTH_THRESHOLD = 0.1  # Minimum depth in meters to consider valid
MAX_DEPTH_THRESHOLD = 3.0  # Maximum depth in meters to consider valid
MOVEMENT_THRESHOLD = 50.0  # Minimum movement in mm to trigger robot motion

# Global variables for shared state
depth_frame_global = None
intrinsics_global = None
robot_dll = None
robot_initialized = False
yolo_model = None
tracking_thread = None
tracking_enabled = False
last_tracked_position = None
tracking_lock = threading.Lock()
pipeline = None  # RealSense pipeline

# Robot control functions
def initialize_robot_control():
    """Initialize the C++ robot control DLL"""
    global robot_dll, robot_initialized
    
    try:
        dll_path = './robot_control.dll'
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found at {dll_path}")
        
        print(f"Loading C++ DLL from: {os.path.abspath(dll_path)}")
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

def move_robot_to_point(x_mm, y_mm, z_mm):
    """Move robot using C++ DLL function"""
    if not robot_initialized or robot_dll is None:
        print("Robot control not available - C++ DLL not loaded or robot not initialized")
        return False
    
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Moving robot to offset: ({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f}) mm")
        
        # Call the C++ function to move robot by offset
        result = robot_dll.move_robot_to_offset(c_float(x_mm), c_float(y_mm), c_float(z_mm))
        
        if result == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ✓ Robot movement completed successfully")
            return True
        else:
            error_messages = {
                -1: "Robot not initialized or error occurred",
                -2: "Robot movement timeout",
                101: "Joint limits or safety constraints",
                112: "Position out of reach"
            }
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ✗ Robot movement failed: {error_messages.get(result, f'Error code {result}')}")
            return False
        
    except Exception as e:
        print(f"Error calling C++ robot function: {e}")
        return False

def initialize_yolo():
    """Initialize YOLO model for object detection"""
    global yolo_model
    
    try:
        print("Loading YOLOv8 model...")
        # Force download of a fresh model
        yolo_model = YOLO('yolov8n.pt')  # This will download the model automatically on first run
        
        # Set model to eval mode to avoid training-related errors
        yolo_model.eval()
        
        print("✓ YOLOv8 model loaded successfully")
        
        # Print available object classes
        print("\nAvailable object classes:")
        for idx, name in yolo_model.names.items():
            print(f"  {idx}: {name}")
        print(f"\nTarget object set to: '{TARGET_LABEL}'")
        
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        print("Attempting to download fresh model...")
        try:
            # Try to force a fresh download
            import os
            if os.path.exists('yolov8n.pt'):
                os.remove('yolov8n.pt')
            yolo_model = YOLO('yolov8n.pt')
            yolo_model.eval()
            print("✓ Fresh YOLOv8 model downloaded and loaded")
        except Exception as e2:
            print(f"✗ Failed to download fresh model: {e2}")
            yolo_model = None

def detect_target_object(color_image, depth_frame, intrinsics):
    """Detect the target object and return its 3D position"""
    if yolo_model is None:
        return None, color_image
    
    try:
        # Run YOLO detection
        results = yolo_model(color_image, verbose=False)
    except Exception as e:
        # Return original image if detection fails
        return None, color_image
    
    target_detection = None
    annotated_image = color_image.copy()
    
    # Process detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = yolo_model.names[class_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check if this is our target object
                if class_name == TARGET_LABEL and confidence > CONFIDENCE_THRESHOLD:
                    # Get depth at center point
                    depth = depth_frame.get_distance(center_x, center_y)
                    
                    # Validate depth
                    if MIN_DEPTH_THRESHOLD < depth < MAX_DEPTH_THRESHOLD:
                        # Deproject to 3D coordinates
                        depth_point = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], depth)
                        
                        # Convert to millimeters
                        obj_x = depth_point[0] * 1000
                        obj_y = depth_point[1] * 1000
                        obj_z = depth_point[2] * 1000
                        
                        # Update target detection if this has higher confidence
                        if target_detection is None or confidence > target_detection['confidence']:
                            target_detection = {
                                'position': (obj_x, obj_y, obj_z),
                                'confidence': confidence,
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y)
                            }
                    
                    # Draw target object in green
                    color = (0, 255, 0)
                else:
                    # Draw other objects in gray
                    color = (128, 128, 128)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Highlight target object if found
    if target_detection:
        x1, y1, x2, y2 = target_detection['bbox']
        center_x, center_y = target_detection['center']
        
        # Draw thick border
        cv2.rectangle(annotated_image, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 3)
        
        # Draw crosshair at center
        cv2.circle(annotated_image, (center_x, center_y), 8, (0, 0, 255), -1)
        cv2.circle(annotated_image, (center_x, center_y), 12, (255, 255, 255), 2)
        cv2.line(annotated_image, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 255), 2)
        cv2.line(annotated_image, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 255), 2)
        
        # Display coordinates
        pos = target_detection['position']
        coord_text = f"Target: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})mm"
        cv2.putText(annotated_image, coord_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return target_detection, annotated_image

def tracking_loop():
    """Background thread for continuous object tracking"""
    global tracking_enabled, last_tracked_position, pipeline
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting tracking thread...")
    
    while tracking_enabled:
        try:
            # Check if pipeline is initialized
            if pipeline is None:
                time.sleep(0.1)
                continue
                
            with tracking_lock:
                if depth_frame_global is None or intrinsics_global is None:
                    time.sleep(0.1)
                    continue
                
                # Get current frames
                depth_frame = depth_frame_global
                intrinsics = intrinsics_global
            
            # Capture current color frame
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except Exception as e:
                print(f"Pipeline frame wait error: {e}")
                time.sleep(0.1)
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                time.sleep(0.1)
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect target object
            detection, _ = detect_target_object(color_image, depth_frame, intrinsics)
            
            if detection:
                new_position = detection['position']
                
                # Check if we should move the robot
                should_move = False
                if last_tracked_position is None:
                    should_move = True
                else:
                    # Calculate distance from last position
                    dx = new_position[0] - last_tracked_position[0]
                    dy = new_position[1] - last_tracked_position[1]
                    dz = new_position[2] - last_tracked_position[2]
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if distance > MOVEMENT_THRESHOLD:
                        should_move = True
                        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Object moved {distance:.1f}mm")
                
                if should_move:
                    # Map camera coordinates to robot coordinates
                    robot_x = new_position[0]
                    robot_y = new_position[1]
                    robot_z = new_position[2]
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Tracking '{TARGET_LABEL}' at ({robot_x:.1f}, {robot_y:.1f}, {robot_z:.1f})mm")
                    
                    # Move robot
                    if move_robot_to_point(robot_x, robot_y, robot_z):
                        last_tracked_position = new_position
            else:
                if last_tracked_position is not None:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Target '{TARGET_LABEL}' lost")
                last_tracked_position = None
            
            # Wait before next tracking update
            time.sleep(TRACKING_INTERVAL)
            
        except Exception as e:
            print(f"Tracking error: {e}")
            time.sleep(0.5)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tracking thread stopped")

def toggle_tracking():
    """Toggle real-time tracking on/off"""
    global tracking_enabled, tracking_thread
    
    if not ENABLE_TRACKING:
        print("Tracking is disabled in configuration")
        return
    
    tracking_enabled = not tracking_enabled
    
    if tracking_enabled:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Real-time tracking ENABLED for '{TARGET_LABEL}'")
        tracking_thread = threading.Thread(target=tracking_loop, daemon=True)
        tracking_thread.start()
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Real-time tracking DISABLED")

def mouse_callback(event, px, py, flags, param):
    """Handle mouse clicks for manual control"""
    global depth_frame_global, intrinsics_global
    
    if depth_frame_global is None or intrinsics_global is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # Get depth value at the pixel
        depth = depth_frame_global.get_distance(px, py)
        
        if depth == 0:
            print("No depth data available at this point.")
            return
        
        # Deproject pixel to 3D point
        depth_point = rs.rs2_deproject_pixel_to_point(intrinsics_global, [px, py], depth)
        
        # Convert to millimeters
        x_mm = depth_point[0] * 1000
        y_mm = depth_point[1] * 1000
        z_mm = depth_point[2] * 1000
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Manual click at camera coordinates: ({x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f})mm")
        
        # Move robot to the clicked position
        move_robot_to_point(x_mm, y_mm, z_mm)

# Initialize components
print("=== RealSense Object Tracking System ===")
print(f"Target Object: {TARGET_LABEL}")
print(f"Tracking Enabled: {ENABLE_TRACKING}")
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
print(f"Movement Threshold: {MOVEMENT_THRESHOLD}mm\n")

initialize_robot_control()
initialize_yolo()

# Configure RealSense pipeline
try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    print("✓ RealSense camera initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize RealSense camera: {e}")
    print("Please ensure the Intel RealSense camera is connected.")
    sys.exit(1)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Create OpenCV window
cv2.namedWindow('RealSense Object Tracking')
cv2.setMouseCallback('RealSense Object Tracking', mouse_callback)

print("\nControls:")
print("- Press 'T' to toggle real-time tracking")
print("- Click on image to manually move robot to that point")
print("- Press 'Q' to quit")
print("- Press 'S' to print tracking status")
print("\nStarting camera feed...\n")

# Automatically start tracking if enabled
if ENABLE_TRACKING and robot_initialized:
    # Small delay to ensure everything is initialized
    time.sleep(1.0)
    print(f"Auto-starting tracking for '{TARGET_LABEL}'...")
    toggle_tracking()

# Main loop
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

        # Update globals for callbacks and tracking
        with tracking_lock:
            depth_frame_global = depth_frame
            intrinsics_global = color_frame.profile.as_video_stream_profile().intrinsics

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Perform object detection and annotation
        detection, annotated_image = detect_target_object(color_image, depth_frame, intrinsics_global)

        # Add status overlay
        status_text = f"Tracking: {'ON' if tracking_enabled else 'OFF'} | Target: {TARGET_LABEL}"
        cv2.putText(annotated_image, status_text, (10, annotated_image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display the annotated image
        cv2.imshow('RealSense Object Tracking', annotated_image)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('t') or key == ord('T'):
            toggle_tracking()
        elif key == ord('s') or key == ord('S'):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status:")
            print(f"  - Tracking: {'ENABLED' if tracking_enabled else 'DISABLED'}")
            print(f"  - Target: {TARGET_LABEL}")
            print(f"  - Last position: {last_tracked_position}")
            print()

finally:
    # Cleanup
    tracking_enabled = False
    if tracking_thread:
        tracking_thread.join(timeout=2.0)
    
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Cleanup robot connection
    if robot_initialized and robot_dll is not None:
        try:
            robot_dll.cleanup_robot()
            print("✓ Robot cleanup completed")
        except Exception as e:
            print(f"Robot cleanup error: {e}")
    
    print("\nSystem shutdown complete.")