# Project Files Overview

## Essential Files for Running realsense_feed.py

### Main Application
- **realsense_feed.py** - Main Python application with object detection and robot control

### Dependencies
- **requirements.txt** - Python package dependencies list
- **yolov8n.pt** - YOLOv8 nano model for object detection

### Robot Control
- **robot_control.dll** - C++ library for controlling the FR5 robot
- **cobotAPI.dll** - FR5 robot SDK library
- **test_move.cpp** - Source code for robot_control.dll (kept for reference/rebuilding)

### Build Scripts
- **build_test_move.bat** - Script to rebuild robot_control.dll if needed

### Documentation
- **README.md** - Original project documentation
- **OBJECT_TRACKING_README.md** - Documentation for the object tracking features

## Running the Application

1. Ensure all Python dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python realsense_feed.py
   ```

## Configuration

Edit these variables at the top of `realsense_feed.py`:
- `TARGET_LABEL = "bottle"` - Object to track
- `ENABLE_TRACKING = True` - Enable/disable automatic tracking
- `CONFIDENCE_THRESHOLD = 0.4` - Detection confidence threshold
- `MOVEMENT_THRESHOLD = 50.0` - Movement threshold in mm