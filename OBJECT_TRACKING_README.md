# RealSense Object Tracking System

## Overview
This system upgrades the existing RealSense + FR5 robot control system with real-time object detection and tracking capabilities using YOLOv8.

## New Features

### 1. Real-Time Object Detection
- Uses YOLOv8n (nano) model for fast object detection
- Detects all COCO dataset objects (80 classes)
- Highlights target object in green, others in gray

### 2. Configurable Target Tracking
- **TARGET_LABEL**: Set the object to track (e.g., "bottle", "person", "cup")
- **ENABLE_TRACKING**: Boolean to enable/disable automatic tracking
- **CONFIDENCE_THRESHOLD**: Minimum confidence for detection (default: 0.4)
- **MOVEMENT_THRESHOLD**: Minimum movement in mm to trigger robot motion (default: 50mm)

### 3. Real-Time Tracking Loop
- Runs in a separate thread to maintain GUI responsiveness
- Continuously tracks the target object as it moves
- Only moves robot when object moves beyond threshold

### 4. Depth Validation
- **MIN_DEPTH_THRESHOLD**: 0.1 meters (filters out noise)
- **MAX_DEPTH_THRESHOLD**: 3.0 meters (filters out far objects)

### 5. Enhanced Logging
- Timestamped logs with millisecond precision
- Tracks object positions and robot movements
- Error messages for failed movements

## Configuration

Edit these variables at the top of `realsense_feed.py`:

```python
TARGET_LABEL = "bottle"      # Object to track
ENABLE_TRACKING = True       # Enable/disable tracking
CONFIDENCE_THRESHOLD = 0.4   # Detection confidence
TRACKING_INTERVAL = 0.5      # Seconds between updates
MOVEMENT_THRESHOLD = 50.0    # Movement threshold in mm
```

## Controls

- **T**: Toggle real-time tracking on/off
- **S**: Show current tracking status
- **Click**: Manual control - move robot to clicked point
- **Q**: Quit application

## Architecture

### Modular Components:
1. **Object Detection Module** (`detect_target_object`)
   - Runs YOLO inference
   - Filters for target object
   - Calculates 3D coordinates using depth

2. **Robot Control Module** (`move_robot_to_point`)
   - Interfaces with C++ DLL
   - Handles coordinate transformation
   - Provides error feedback

3. **Tracking Thread** (`tracking_loop`)
   - Runs continuously in background
   - Monitors target object position
   - Triggers robot movement when needed

4. **GUI Thread** (main loop)
   - Displays annotated video feed
   - Handles user input
   - Remains responsive during tracking

## Error Handling

- Invalid depth readings are filtered out
- Robot movement errors are caught and logged
- Thread-safe access to shared camera data
- Graceful shutdown on exit

## Requirements

- Intel RealSense D435i camera
- FR5 robot connected at 192.168.58.2
- Python packages: pyrealsense2, numpy, opencv-python, ultralytics
- C++ robot control DLL (robot_control.dll)

## Usage

1. Ensure robot is in automatic mode
2. Run: `python realsense_feed.py`
3. Press 'T' to start tracking the configured object
4. Place target object in camera view
5. Robot will follow the object automatically

## Common Issues

- **"No depth data"**: Object too close/far or reflective surface
- **"Position out of reach"**: Object outside robot workspace
- **"Joint limits"**: Robot cannot reach position safely
- **Object not detected**: Adjust CONFIDENCE_THRESHOLD or lighting

## Performance Notes

- YOLOv8n provides ~30 FPS on modern CPUs
- Tracking updates every 0.5 seconds (configurable)
- Movement threshold prevents jittery motion
- Threading ensures smooth GUI operation