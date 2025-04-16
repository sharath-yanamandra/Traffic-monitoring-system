# config.py
class Config:
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    
    # Video settings
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Model paths
    MODEL_PATHS = {
        'accident': 'models/accident_detection.pt',
        'helmet': 'models/helmet_detection.pt',
        'people_animals': 'models/people_animals_detection.pt',
        'speed': 'models/speed_detection.pt',
        'triple_riding': 'models/triple_riding_detection.pt',
        'wrong_lane': 'models/wrong_lane_detection.pt',
        'vehicle': 'models/vehicle_monitoring.pt'
    }
    
    # Database settings
    DB_HOST = 'localhost'
    DB_PORT = 27017
    DB_NAME = 'traffic_monitoring'
    
    # Alert settings
    ALERT_TIMEOUT = 30  # seconds
    
# utils.py
import cv2
import numpy as np
from typing import Tuple, List, Dict
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_video_writer(output_path: str, frame_width: int, frame_height: int, fps: int) -> cv2.VideoWriter:
    """Initialize video writer for saving processed frames"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def draw_detections(frame: np.ndarray, detections: List[Dict], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes and labels on frame"""
    frame_copy = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = detection['class']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f'{label}: {confidence:.2f}'
        cv2.putText(frame_copy, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame_copy

def calculate_intersection_over_union(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def save_violation_record(violation_type: str, frame: np.ndarray, timestamp: datetime, location: str):
    """Save violation record to database and disk"""
    # Generate unique filename
    filename = f"{violation_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    
    # Save frame to disk
    cv2.imwrite(f"violations/{filename}", frame)
    
    # Create violation record
    record = {
        'type': violation_type,
        'timestamp': timestamp,
        'location': location,
        'image_path': filename
    }
    
    # Save to database (implement database connection as needed)
    logger.info(f"Violation recorded: {violation_type} at {timestamp}")
    return record

def generate_alert(violation_type: str, detection: Dict, frame: np.ndarray):
    """Generate alert for detected violation"""
    alert = {
        'type': violation_type,
        'timestamp': datetime.now(),
        'confidence': detection['confidence'],
        'location': detection['bbox'],
        'frame': frame
    }
    
    # Log alert
    logger.warning(f"Alert generated: {violation_type}")
    return alert
