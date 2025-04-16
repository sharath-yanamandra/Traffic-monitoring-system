import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

class TripleRidingDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Constants
        self.MOTORCYCLE_CLASS = 3  # COCO class for motorcycle
        self.PERSON_CLASS = 0      # COCO class for person
        
        # Colors for visualization
        self.colors = {
            'normal': (0, 255, 0),    # Green
            'violation': (0, 0, 255)   # Red
        }
        
        # Tracking data
        self.track_history = defaultdict(list)
        self.rider_counts = defaultdict(int)
        self.detection_persistence = 5  # frames
    
    def detect(self, frame):
        """Detect triple riding in a frame"""
        stats = {
            'total_motorcycles': 0,
            'triple_riding': 0,
            'normal_riding': 0
        }
        
        try:
            # Run tracking
            results = self.model.track(frame, persist=True, conf=0.5,
                                     classes=[self.MOTORCYCLE_CLASS, self.PERSON_CLASS])
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Separate motorcycles and persons
                motorcycles = []
                motorcycle_ids = []
                persons = []
                
                for box, cls, track_id in zip(boxes, classes, track_ids):
                    if cls == self.MOTORCYCLE_CLASS:
                        motorcycles.append(box)
                        motorcycle_ids.append(track_id)
                        stats['total_motorcycles'] += 1
                    elif cls == self.PERSON_CLASS:
                        persons.append(box)
                
                # Process each motorcycle
                for motorcycle_box, motorcycle_id in zip(motorcycles, motorcycle_ids):
                    # Expand box to better capture riders
                    expanded_box = self._expand_box(motorcycle_box, frame.shape)
                    
                    # Count riders
                    rider_count = 0
                    for person_box in persons:
                        if self._calculate_overlap(expanded_box, person_box) > 0.3:
                            rider_count += 1
                    
                    # Update tracking history
                    self.track_history[motorcycle_id].append(rider_count)
                    if len(self.track_history[motorcycle_id]) > self.detection_persistence:
                        self.track_history[motorcycle_id].pop(0)
                    
                    # Calculate smooth rider count
                    smooth_count = max(1, round(np.mean(self.track_history[motorcycle_id])))
                    self.rider_counts[motorcycle_id] = smooth_count
                    
                    # Update statistics
                    if smooth_count >= 3:
                        stats['triple_riding'] += 1
                    else:
                        stats['normal_riding'] += 1
                    
                    # Draw detection
                    is_violation = smooth_count >= 3
                    color = self.colors['violation'] if is_violation else self.colors['normal']
                    text = "TRIPLE RIDING!" if is_violation else f"Riders: {smooth_count}"
                    self._draw_detection(frame, motorcycle_box, text, color)
                
                # Clean up old tracks
                for track_id in list(self.track_history.keys()):
                    if track_id not in motorcycle_ids:
                        del self.track_history[track_id]
                        if track_id in self.rider_counts:
                            del self.rider_counts[track_id]
            
            # Draw statistics and timestamp
            self._draw_stats(frame, stats)
            self._add_timestamp(frame)
            
            return {
                'stats': stats,
                'frame': frame
            }
            
        except Exception as e:
            print(f"Error in triple riding detection: {str(e)}")
            return {
                'stats': stats,
                'frame': frame
            }
    
    def _expand_box(self, box, frame_shape, expand_ratio=1.3):
        """Expand bounding box to better capture riders"""
        height, width = frame_shape[:2]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        box_width = (box[2] - box[0]) * expand_ratio
        box_height = (box[3] - box[1]) * expand_ratio
        
        x1 = max(0, center_x - box_width/2)
        y1 = max(0, center_y - box_height/2)
        x2 = min(width, center_x + box_width/2)
        y2 = min(height, center_y + box_height/2)
        
        return [x1, y1, x2, y2]
    
    def _calculate_overlap(self, box1, box2):
        """Calculate overlap between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / min(area1, area2)
    
    def _draw_detection(self, frame, box, text, color):
        """Draw detection box with label"""
        x1, y1, x2, y2 = map(int, box)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        cv2.putText(frame, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_stats(self, frame, stats):
        """Draw statistics on frame"""
        y_pos = 30
        texts = [
            f"Total Motorcycles: {stats['total_motorcycles']}",
            f"Triple Riding: {stats['triple_riding']}",
            f"Normal Riding: {stats['normal_riding']}"
        ]
        
        for text in texts:
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (255, 255, 255), 2)
            y_pos += 30
    
    def _add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)