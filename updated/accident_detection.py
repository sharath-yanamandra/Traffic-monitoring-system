import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class AccidentDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Define colors for visualization
        self.colors = {
            'normal': (0, 255, 0),     # green for normal traffic
            'accident': (0, 0, 255)     # red for accidents
        }
        
        # Define accident-related classes
        self.accident_classes = [1, 2, 3, 5, 6, 7, 8]  # classes that indicate accidents
        
    def detect(self, frame):
        """Detect accidents in a frame"""
        stats = {
            'accident_detected': False,
            'accident_count': 0,
            'normal_count': 0
        }
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=0.5)
            
            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if it's an accident class
                    is_accident = cls in self.accident_classes
                    
                    # Update statistics
                    if is_accident:
                        stats['accident_count'] += 1
                        stats['accident_detected'] = True
                    else:
                        stats['normal_count'] += 1
                    
                    # Draw detection box
                    color = self.colors['accident'] if is_accident else self.colors['normal']
                    self._draw_detection(frame, [x1, y1, x2, y2], 
                                      "Accident" if is_accident else "Normal", 
                                      conf, color)
            
            # Draw warning if accident detected
            if stats['accident_detected']:
                cv2.putText(frame, "ACCIDENT DETECTED!", 
                           (frame.shape[1]//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add timestamp
            self._add_timestamp(frame)
            
            return {
                'stats': stats,
                'frame': frame
            }
            
        except Exception as e:
            print(f"Error in accident detection: {str(e)}")
            return {
                'stats': stats,
                'frame': frame
            }
    
    def _draw_detection(self, frame, box, label, conf, color):
        """Draw detection box with label"""
        x1, y1, x2, y2 = map(int, box)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f'{label} {conf:.2f}'
        cv2.putText(frame, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)