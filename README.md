# Traffic Monitoring System

## Overview

The Traffic Monitoring System is a comprehensive computer vision-based solution for monitoring and analyzing traffic conditions using YOLO (You Only Look Once) object detection models. The system processes video streams from traffic cameras in real-time to detect various traffic violations, monitor vehicle flows, and enhance overall road safety.

## Features

The system offers several detection and monitoring capabilities:

- **Accident Detection**: Identifies vehicle accidents and provides alerts in real-time
- **Helmet Detection**: Monitors motorcyclists for helmet compliance
- **Speed Detection**: Measures vehicle speeds and flags speed limit violations
- **Triple Riding Detection**: Identifies motorcycles with more than two riders
- **Wrong Lane Detection**: Detects vehicles traveling in the wrong direction
- **Vehicle Monitoring**: Counts and classifies vehicles by type (car, motorcycle, bus, truck)
- **People and Animals Detection**: Identifies pedestrians and animals on roadways

## System Architecture

### Core Components

- **Main System (`main.py`)**: Coordinates all detection modules and manages camera streams
- **Detection Modules**:
  - `accident_detection.py`
  - `helmet_detection.py`
  - `speed_detection.py`
  - `triple_riding_detection.py`
  - `wrong_lane_detection.py`
  - `vehicle_monitoring.py`
  - `people_and_animals.py`
- **Web Interface**: Flask-based API and WebSocket for real-time visualization

### Technology Stack

- **Computer Vision**: OpenCV, YOLO (Ultralytics)
- **Backend**: Python, Flask
- **Real-time Communication**: Flask-SocketIO
- **Frontend Support**: Base64 encoding for image transmission

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLO
- Flask
- Flask-SocketIO
- Flask-CORS
- NumPy
- Additional dependencies as specified in the code

## Installation

1. Clone the repository to your local machine
2. Install the required dependencies:
   ```bash
   pip install opencv-python ultralytics flask flask-socketio flask-cors numpy
   ```
3. Download the required YOLO weights files and place them in the `weights` directory:
   - `weights/helmet_weight.pt`
   - `weights/accident_weight.pt`
   - `weights/speed_weight.pt`
   - `weights/triple_riding_weight.pt`
   - `weights/wrong_lane_weight.pt`
   - `weights/vehicle_weight.pt`
   - `weights/yolov8x.pt` (for people and animals detection)

## Usage

### Starting the System

1. Run the main script:
   ```bash
   python main.py
   ```
2. The Flask server will start on port 5000 by default (http://localhost:5000)

### API Endpoints

- **Start Camera Stream**:
  ```
  POST /api/start_stream
  Body: {"camera_id": "cam1", "rtsp_url": "rtsp://example.com/stream"}
  ```

- **Stop Camera Stream**:
  ```
  POST /api/stop_stream
  Body: {"camera_id": "cam1"}
  ```

- **Get Current Statistics**:
  ```
  GET /api/statistics
  ```

### WebSocket Events

- Connect to the socket instance to receive real-time detection results
- Events are emitted with the format: `detection_results_{camera_id}`

## Detection Modules

### Accident Detection

The `AccidentDetector` class processes video frames to identify potential accident scenarios based on pre-trained YOLO models.

```python
from accident_detection import AccidentDetector

detector = AccidentDetector('weights/accident_weight.pt')
results = detector.detect(frame)
```

### Helmet Detection

The `HelmetDetector` identifies motorcyclists and classifies them based on helmet usage.

```python
from helmet_detection import HelmetDetector

detector = HelmetDetector('weights/helmet_weight.pt')
results = detector.detect(frame)
```

### Speed Detection

The `SpeedDetector` calculates vehicle speeds based on frame-to-frame movement and real-world distance calibration.

```python
from speed_detection import SpeedDetector

detector = SpeedDetector('weights/speed_weight.pt', real_world_distance=20, speed_limit=40)
detector.set_region([(100, 300), (500, 300)])
results = detector.detect(frame)
```

### Triple Riding Detection

The `TripleRidingDetector` identifies motorcycles carrying more than two people.

```python
from triple_riding_detection import TripleRidingDetector

detector = TripleRidingDetector('weights/triple_riding_weight.pt')
results = detector.detect(frame)
```

### Wrong Lane Detection

The `WrongLaneDetector` tracks vehicles and determines if they are traveling in the wrong direction.

```python
from wrong_lane_detection import WrongLaneDetector

detector = WrongLaneDetector('weights/wrong_lane_weight.pt')
results = detector.detect(frame)
```

### Vehicle Monitoring

The `VehicleMonitor` counts and classifies vehicles by type and direction.

```python
from vehicle_monitoring import VehicleMonitor

monitor = VehicleMonitor('weights/vehicle_weight.pt')
monitor.set_counting_line(350)
results = monitor.detect(frame)
```

### People and Animals Detection

The `PeopleAndAnimalsDetector` identifies pedestrians and animals on roadways.

```python
from people_and_animals import PeopleAndAnimalsDetector

detector = PeopleAndAnimalsDetector('weights/yolov8x.pt')
results = detector.detect(frame)
```

## Customization

### Adjusting Detection Parameters

- Modify confidence thresholds in each detector class
- Adjust region of interest for speed detection
- Set custom counting lines for vehicle monitoring
- Define custom class mappings for detection models

### Adding New Detection Modules

1. Create a new Python file for your detector following the existing pattern
2. Implement a class with at least an `__init__` and `detect` method
3. Register the detector in the `TrafficMonitoringSystem.initialize_models` method

## Output Format

Each detector returns a dictionary with:
- `stats`: Detection statistics specific to the module
- `frame`: Processed frame with visualizations

## License

[Insert License Information]

## Contributors

[sharath yanamandra]

## Acknowledgements

- YOLOv8 by Ultralytics
- OpenCV community
