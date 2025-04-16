import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class HelmetDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Define colors for visualization
        self.colors = {
            'With_Helmet': (0, 255, 0),    # Green
            'Without_Helmet': (0, 0, 255)   # Red
        }
        
    def detect(self, frame):
        """Detect helmets in a frame"""
        stats = {
            'With_Helmet': 0,
            'Without_Helmet': 0
        }
        
        try:
            # Run detection
            results = self.model(frame, conf=0.5)
            
            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Get class and confidence
                    cls = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    # Update statistics
                    stats[cls] += 1
                    
                    # Draw detection box
                    self._draw_detection(frame, [x1, y1, x2, y2], cls, conf)
            
            # Draw statistics and timestamp
            self._draw_stats(frame, stats)
            self._add_timestamp(frame)
            
            return {
                'stats': stats,
                'frame': frame
            }
            
        except Exception as e:
            print(f"Error in helmet detection: {str(e)}")
            return {
                'stats': stats,
                'frame': frame
            }
    
    def _draw_detection(self, frame, box, cls, conf):
        """Draw detection box with label"""
        x1, y1, x2, y2 = map(int, box)
        color = self.colors[cls]
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f'{cls} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                   
    def _draw_stats(self, frame, stats):
        """Draw statistics on frame"""
        y_pos = 30
        for cls, count in stats.items():
            text = f'{cls}: {count}'
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       self.colors[cls], 2)
            y_pos += 30
    
    def _add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)