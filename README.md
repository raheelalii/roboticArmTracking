# FR5 RealSense Object Detection Web App

This project provides a simple web application that displays a live view from
an Intel® RealSense™ D457 depth camera, runs object detection on the
colour stream, and allows a user to command a FAIRINO FR5 robot arm to
move its tool centre point to the centre of a detected object.  The
code is organised into three main packages:

* **camera/** – wraps the RealSense camera and implements object
  detection using OpenCV’s DNN module.  You need to supply YOLO model
  files (`.cfg`, `.weights` and `.names`) in `camera/models`.
* **robot/** – contains a lightweight Python client for the FR5’s
  REST API.  You may need to adjust the endpoint path and payload to
  match your controller’s firmware.
* **web/** – a Flask application that ties everything together.  It
  exposes a web page with a live MJPEG stream and a simple form to
  specify an object to move to.

## Setup

1. **Install dependencies.**  Create a Python virtual environment
   (optional) and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   In addition you must install the Intel RealSense SDK (`pyrealsense2`) for
   your operating system.  See [Intel’s instructions](https://www.intelrealsense.com/developers/) for details.

2. **Download YOLO model files.**  The object detector uses OpenCV’s
   Darknet interface.  Place a YOLO configuration file (e.g.
   `yolov4-tiny.cfg`), corresponding weights file (e.g.
   `yolov4-tiny.weights`) and class names file (e.g. `coco.names`) into
   `camera/models`.  You can obtain these files from the official
   [Darknet repository](https://github.com/pjreddie/darknet) or other sources.

3. **Configure the robot endpoint.**  The FR5 controller’s API may
   expose different endpoints depending on your firmware.  Edit
   `robot/robot.py` if necessary.  By default the client posts to
   `http://192.168.58.2:80/api/move_to_cartesian`.

## Running the Application

Start the Flask development server with:

```bash
export FLASK_APP=web/app.py
flask run --host=0.0.0.0 --port=5000
```

Open a web browser and navigate to `http://localhost:5000/`.  The
page will display the live video feed.  Enter the name of an object
listed in `coco.names` (e.g. “bottle”, “cup”) and click **Move to
Object**.  The application will locate the object in the most recent
frame, compute its 3‑D coordinates and send a move command to the
robot arm.

> **Warning:** Moving a robot to automatically detected points can be
> dangerous.  Always ensure the working area is clear and monitor the
> robot’s motion.  Use conservative speeds and verify the endpoint
> payload in `robot/robot.py` matches your robot’s API.