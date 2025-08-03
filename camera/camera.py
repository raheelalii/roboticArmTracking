"""
camera/camera.py

This module contains helper classes to interface with an Intel® RealSense™
depth camera and perform object detection using OpenCV’s deep‑learning (DNN)
module.  It is organised around two core classes:

* ``RealSenseCamera`` – encapsulates the RealSense pipeline, exposing
  colour and depth frames along with a convenience method to convert
  2‑D image coordinates into 3‑D space.
* ``ObjectDetector`` – wraps a YOLO model for object detection.  The
  detector loads a Darknet configuration and weight file and provides a
  ``detect()`` method that returns a list of detections.  It uses
  OpenCV’s ``cv2.dnn.DetectionModel`` which supports CPU and CUDA backends.

If you wish to draw the detections on an image, the helper function
``draw_detections()`` is provided.

The RealSense SDK (``pyrealsense2``) is only imported when required;
if the module is missing an informative ``ImportError`` is raised.

Note: You must supply your own YOLO model files (``.cfg``, ``.weights`` and
``.names``) in the ``camera/models`` directory.  These files are not
distributed with this code to keep the repository size reasonable.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

# Attempt to import the RealSense SDK.  If it's not installed the
# RealSenseCamera class will raise an ImportError during initialisation.
try:
    import pyrealsense2 as rs  # type: ignore
except ImportError:
    rs = None  # RealSense is optional until actually used


class RealSenseCamera:
    """Wrapper around an Intel® RealSense™ camera.

    The class starts a pipeline streaming both colour and depth frames.  It
    performs alignment so that depth pixels correspond to the colour frame.
    It also exposes camera intrinsics and depth scale so callers can
    convert 2‑D pixel coordinates into 3‑D coordinates in metres.

    Args:
        width: Width of the colour/depth streams (default 640).
        height: Height of the colour/depth streams (default 480).
        fps: Frames per second to request from the camera (default 30).

    Raises:
        ImportError: If ``pyrealsense2`` is not installed.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        if rs is None:
            raise ImportError(
                "pyrealsense2 is required to use RealSenseCamera. Install the Intel RealSense SDK."
            )
        # Start a RealSense pipeline
        self.pipeline: rs.pipeline = rs.pipeline()
        self.config: rs.config = rs.config()
        # Enable colour and depth streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # Align depth to colour frame
        self.align = rs.align(rs.stream.color)
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        # Depth scale used to convert raw depth values to metres
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale: float = self.depth_sensor.get_depth_scale()
        # Cache camera intrinsics for deprojection
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics: rs.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retrieve a single colour/depth frame pair.

        Returns:
            A tuple ``(color_image, depth_image)`` where each element is a
            NumPy array.  If frames are not available ``(None, None)`` is returned.
        """
        frames = self.pipeline.wait_for_frames()
        # Align depth to colour
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        depth_image: np.ndarray = np.asanyarray(depth_frame.get_data())
        color_image: np.ndarray = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def deproject(self, pixel: Tuple[int, int], depth_value: float) -> Tuple[float, float, float]:
        """Convert a 2‑D pixel and its depth value into a 3‑D point in metres.

        Args:
            pixel: ``(x, y)`` pixel coordinates in the colour image.
            depth_value: Raw depth value at that pixel (from a depth frame).

        Returns:
            A tuple ``(x, y, z)`` representing the 3‑D point in metres.

        Raises:
            ImportError: If the RealSense SDK is unavailable.
        """
        if rs is None:
            raise ImportError(
                "pyrealsense2 must be installed to deproject pixels."
            )
        depth_meters = float(depth_value) * self.depth_scale
        # Deproject to 3D coordinates using camera intrinsics
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(pixel[0]), float(pixel[1])], depth_meters)
        # rs2_deproject_pixel_to_point returns in metres; convert to tuple
        return float(point[0]), float(point[1]), float(point[2])

    def stop(self) -> None:
        """Stop the camera pipeline and release resources."""
        self.pipeline.stop()


class ObjectDetector:
    """Object detector using a YOLO model via OpenCV's DNN module.

    The detector loads network architecture and weights from Darknet files
    (``.cfg`` and ``.weights``) and a list of class names.  It then
    exposes a ``detect()`` method returning a list of detections.  Each
    detection contains the class ID, label, confidence and bounding box.

    Args:
        cfg_path: Path to the YOLO configuration file.
        weights_path: Path to the YOLO weights file.
        names_path: Path to a file containing class names (one per line).
        input_size: Model input resolution (width, height), typically (416, 416).
        conf_threshold: Minimum confidence threshold for detections.
        nms_threshold: Non‑maximum suppression threshold to filter overlapping boxes.
    """

    def __init__(
        self,
        cfg_path: str,
        weights_path: str,
        names_path: str,
        input_size: Tuple[int, int] = (416, 416),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        # Validate file paths early and raise informative errors
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"YOLO config file not found: {cfg_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Class names file not found: {names_path}")
        # Load the network from Darknet files
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        # Prefer CUDA if available; fall back to CPU
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Build a detection model wrapper
        self.model = cv2.dnn.DetectionModel(self.net)
        self.model.setInputParams(scale=1 / 255.0, size=input_size, swapRB=True)
        # Load class names
        with open(names_path, 'r', encoding='utf-8') as f:
            self.classes: List[str] = [line.strip() for line in f.readlines() if line.strip()]
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)

    def detect(self, frame: np.ndarray) -> List[Dict[str, object]]:
        """Run object detection on a BGR image.

        Args:
            frame: Input image in BGR format.

        Returns:
            A list of dictionaries describing detected objects.  Each dict
            contains the keys:

            ``id`` (int): Class index.
            ``label`` (str): Human‑readable class name.
            ``confidence`` (float): Confidence score between 0 and 1.
            ``box`` (Tuple[int, int, int, int]): Bounding box (x, y, width, height).
        """
        # Ensure frame is a NumPy array
        if frame is None or not isinstance(frame, np.ndarray):
            return []
        classes_ids, confidences, boxes = self.model.detect(
            frame, confThreshold=self.conf_threshold, nmsThreshold=self.nms_threshold
        )
        detections: List[Dict[str, object]] = []
        for class_id, confidence, box in zip(classes_ids, confidences, boxes):
            # Guard against invalid class indices
            label = self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id)
            detections.append(
                {
                    "id": int(class_id),
                    "label": label,
                    "confidence": float(confidence),
                    "box": tuple(int(v) for v in box),  # x, y, w, h
                }
            )
        return detections


def draw_detections(frame: np.ndarray, detections: List[Dict[str, object]], colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Each detection is drawn as a coloured rectangle with a filled label
    background.  The provided ``colors`` dict may map class names to BGR
    colour tuples; otherwise a default green colour is used.

    Args:
        frame: Image on which to draw.  The image will be modified in place.
        detections: List of detection dicts as returned by ``ObjectDetector.detect``.
        colors: Optional mapping from class label to BGR colour.

    Returns:
        The image with drawings applied (same object that was passed in).
    """
    for det in detections:
        x, y, w, h = det["box"]
        label: str = det["label"]
        confidence: float = det["confidence"]
        # Choose colour for this label
        if colors and label in colors:
            colour = colors[label]
        else:
            # default to green
            colour = (0, 255, 0)
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        # Prepare text
        text = f"{label}: {confidence:.2f}"
        # Compute text size
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Draw background rectangle for text
        cv2.rectangle(frame, (x, y - text_h - baseline), (x + text_w, y), colour, -1)
        # Draw text
        cv2.putText(frame, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame