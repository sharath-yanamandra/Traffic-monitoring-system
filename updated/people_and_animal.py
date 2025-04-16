import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class PeopleAndAnimalsDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Define classes
        self.person_class_id = 0  # Person class
        self.animal_class_ids = [16, 17, 18, 19, 20, 21, 22]  # Bird, cat, dog, horse, sheep, cow, elephant
        
        # Colors for visualization
        self.colors = {
            'person': (0, 255, 0),  # Green for person
            'animal': (255, 0, 0)   # Blue for animals
        }
    
    def detect(self, frame):
        """Detect people and animals in a frame"""
        stats = {
            'person_count': 0,
            'animal_count': 0,
            'detections': {
                'people': [],
                'animals': []
            }
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
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check class type and update stats
                    if cls == self.person_class_id:
                        stats['person_count'] += 1
                        stats['detections']['people'].append({
                            'box': [x1, y1, x2, y2],
                            'confidence': conf
                        })
                        self._draw_detection(frame, [x1, y1, x2, y2], 
                                          'Person', conf, self.colors['person'])
                    
                    elif cls in self.animal_class_ids:
                        stats['animal_count'] += 1
                        stats['detections']['animals'].append({
                            'box': [x1, y1, x2, y2],
                            'confidence': conf
                        })
                        self._draw_detection(frame, [x1, y1, x2, y2],
                                          'Animal', conf, self.colors['animal'])
            
            # Draw statistics and timestamp
            self._draw_stats(frame, stats)
            self._add_timestamp(frame)
            
            return {
                'stats': stats,
                'frame': frame
            }
            
        except Exception as e:
            print(f"Error in people and animals detection: {str(e)}")
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
    
    def _draw_stats(self, frame, stats):
        """Draw statistics on frame"""
        # Draw person count
        cv2.putText(frame, f'Persons Detected: {stats["person_count"]}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, self.colors['person'], 2)
        
        # Draw animal count
        cv2.putText(frame, f'Animals Detected: {stats["animal_count"]}',
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   1, self.colors['animal'], 2)
    
    def _add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp,
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)