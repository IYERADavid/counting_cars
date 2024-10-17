# YOLO Object Detection and Car Tracking

## Overview

This project implements real-time object detection and vehicle tracking using the YOLOv8 model. The system detects cars in video footage, tracks their movement, calculates their speed, and displays relevant information.

## Features

- Detects and tracks vehicles using the YOLOv8 model.
- Calculates vehicle speed based on movement across defined lines.
- Saves the video output with overlays for bounding boxes and speed information.
- Logs car data (speed, object ID, timestamps) to a text file for analysis.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO (Install using `pip install ultralytics`)
- Pandas
- Numpy

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/ZIEDSAGGUEM/counting_cars.git
   cd counting_cars

2. Run the script:

   ```bash
   python speed.py

