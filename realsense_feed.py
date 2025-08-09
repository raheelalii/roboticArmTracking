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
TARGET_LABEL = "bottle"  # Primary object to track
SECOND_TARGET_LABEL = "scissors"  # Secondary target to move the first object to
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
last_scissors_position = None
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

def move_robot_to_point(x_mm, y_mm, z_mm, compensate_approach: bool = True):
    """Move robot using C++ DLL function.
    compensate_approach: if True, add +50 mm on Z to counter a built-in 50 mm approach offset in the bridge,
    so the effective motion reaches the exact camera-read Z.
    """
    if not robot_initialized or robot_dll is None:
        print("Robot control not available - C++ DLL not loaded or robot not initialized")
        return False
    
    try:
        if compensate_approach:
            z_mm = z_mm + 50.0
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
        print(f"\nPrimary target set to: '{TARGET_LABEL}', secondary target: '{SECOND_TARGET_LABEL}'")
        
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

def _deproject_center(depth_frame, intrinsics, cx, cy):
    depth = depth_frame.get_distance(cx, cy)
    if not (MIN_DEPTH_THRESHOLD < depth < MAX_DEPTH_THRESHOLD):
        return None
    pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
    return (pt[0] * 1000.0, pt[1] * 1000.0, pt[2] * 1000.0)  # mm

def detect_bottle_and_scissors(color_image, depth_frame, intrinsics):
    """Detect best 'bottle' and 'scissors' in a single YOLO pass.
    Returns (bottle_detection, scissors_detection, annotated_image)
    Detection dict: {position: (x_mm,y_mm,z_mm), confidence: float, bbox: (x1,y1,x2,y2), center: (u,v)}
    """
    if yolo_model is None:
        return None, None, color_image
    try:
        results = yolo_model(color_image, verbose=False)
    except Exception:
        return None, None, color_image

    best_bottle = None
    best_scissors = None
    annotated = color_image.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            class_name = yolo_model.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            pos_mm = _deproject_center(depth_frame, intrinsics, cx, cy)
            if pos_mm is None:
                color = (128, 128, 128)
            else:
                color = (0, 255, 0) if class_name == TARGET_LABEL else ((255, 0, 255) if class_name == SECOND_TARGET_LABEL else (128, 128, 128))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{class_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if class_name == TARGET_LABEL and pos_mm is not None:
                det = {
                    'position': pos_mm,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy)
                }
                if best_bottle is None or conf > best_bottle['confidence']:
                    best_bottle = det
            elif class_name == SECOND_TARGET_LABEL and pos_mm is not None:
                det = {
                    'position': pos_mm,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': (cx, cy)
                }
                if best_scissors is None or conf > best_scissors['confidence']:
                    best_scissors = det

    # Emphasize centers
    for det, col in [(best_bottle, (0, 255, 0)), (best_scissors, (255, 0, 255))]:
        if det:
            u, v = det['center']
            cv2.circle(annotated, (u, v), 6, col, -1)

    return best_bottle, best_scissors, annotated

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

print("- Press 'M' to move primary target to secondary target (bottle -> scissors)")
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

        # Perform object detection and annotation (both targets)
        bottle_det, scissors_det, annotated_image = detect_bottle_and_scissors(color_image, depth_frame, intrinsics_global)

        # Add status overlay
        status_text = f"Tracking: {'ON' if tracking_enabled else 'OFF'} | Target: {TARGET_LABEL} | Second: {SECOND_TARGET_LABEL}"
        cv2.putText(annotated_image, status_text, (10, annotated_image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show coordinates on screen
        y_text = 30
        if bottle_det:
            bx, by, bz = bottle_det['position']
            cv2.putText(annotated_image, f"Bottle: ({bx:.0f},{by:.0f},{bz:.0f}) mm", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y_text += 25
        if scissors_det:
            sx, sy, sz = scissors_det['position']
            last_scissors_position = scissors_det['position']
            cv2.putText(annotated_image, f"Scissors: ({sx:.0f},{sy:.0f},{sz:.0f}) mm", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            y_text += 25

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
            print(f"  - Primary target: {TARGET_LABEL}")
            print(f"  - Secondary target: {SECOND_TARGET_LABEL}")
            print(f"  - Last bottle position: {last_tracked_position}")
            print(f"  - Last scissors position: {last_scissors_position}")
            print()
        elif key == ord('m') or key == ord('M'):
            # Move bottle -> return to original -> re-detect scissors -> move to scissors
            if not robot_initialized:
                print("[WARN] Robot not initialized.")
                continue
            if not bottle_det:
                print("[WARN] Bottle not visible with valid depth.")
                continue
            bx, by, bz = bottle_det['position']
            # Clamp absolute offset to prevent large jumps
            def clamp(v, lim=350.0):
                return max(-lim, min(lim, v))
            cbx, cby, cbz = clamp(bx), clamp(by), clamp(bz)
            print(f"[PLAN] Step1: go to bottle ({cbx:.1f},{cby:.1f},{cbz:.1f}) mm")
            ok1 = move_robot_to_point(cbx, cby, cbz, compensate_approach=True)
            if not ok1:
                print("[RESULT] Move had errors on bottle approach.")
                continue
            time.sleep(0.5)
            print(f"[PLAN] Step2: return to original by offset ({-cbx:.1f},{-cby:.1f},{-cbz:.1f}) mm")
            ok2 = move_robot_to_point(-cbx, -cby, -cbz, compensate_approach=True)
            if not ok2:
                print("[RESULT] Return to original failed.")
                continue
            time.sleep(0.5)
            # Step3: re-detect scissors from the original pose
            print("[PLAN] Step3: searching for scissors from original pose...")
            found_scissors = None
            for _ in range(45):  # about ~1.5s at 30 FPS
                frames2 = pipeline.wait_for_frames()
                af2 = align.process(frames2)
                df2 = af2.get_depth_frame()
                cf2 = af2.get_color_frame()
                if not df2 or not cf2:
                    continue
                with tracking_lock:
                    intr2 = cf2.profile.as_video_stream_profile().intrinsics
                ci2 = np.asanyarray(cf2.get_data())
                _, sc_det2, _ = detect_bottle_and_scissors(ci2, df2, intr2)
                if sc_det2:
                    found_scissors = sc_det2['position']
                    break
                time.sleep(0.02)
            if not found_scissors:
                print("[WARN] Scissors not found after returning to original pose.")
                continue
            sx, sy, sz = found_scissors
            csx, csy, csz = clamp(sx), clamp(sy), clamp(sz)
            print(f"[PLAN] Step4: move to scissors ({csx:.1f},{csy:.1f},{csz:.1f}) mm")
            ok3 = move_robot_to_point(csx, csy, csz, compensate_approach=True)
            print("[RESULT] Move complete." if ok3 else "[RESULT] Move had errors on scissors approach.")

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