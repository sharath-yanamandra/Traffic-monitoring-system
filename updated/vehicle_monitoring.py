import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

class VehicleMonitor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Define target classes
        self.target_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Colors for visualization
        self.colors = {
            'car': (0, 255, 0),       # Green
            'motorcycle': (255, 0, 0), # Blue
            'bus': (0, 0, 255),       # Red
            'truck': (255, 255, 0)     # Cyan
        }
        
        # Tracking data
        self.vehicle_counts = {
            'left': defaultdict(int),
            'right': defaultdict(int)
        }
        self.counted_vehicles = {
            'left': set(),
            'right': set()
        }
        
        # Line for counting (can be set)
        self.counting_line = None
    
    def set_counting_line(self, y_position):
        """Set the y-position of the counting line"""
        self.counting_line = y_position
    
    def detect(self, frame):
        """Monitor vehicles in a frame"""
        if self.counting_line is None:
            self.counting_line = frame.shape[0] // 2
        
        stats = {
            'total': {'left': 0, 'right': 0},
            'by_type': {
                'left': {cls: 0 for cls in self.target_classes.values()},
                'right': {cls: 0 for cls in self.target_classes.values()}
            }
        }
        
        try:
            # Run tracking
            results = self.model.track(frame, persist=True, conf=0.5,
                                     classes=list(self.target_classes.keys()))
            
            # Draw counting line
            cv2.line(frame, (0, self.counting_line),
                    (frame.shape[1], self.counting_line),
                    (255, 255, 255), 2)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Process each detection
                for box, cls, track_id in zip(boxes, classes, track_ids):
                    if cls not in self.target_classes:
                        continue
                    
                    vehicle_type = self.target_classes[int(cls)]
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Determine direction and count vehicles
                    if self.counting_line - 10 < center_y < self.counting_line + 10:
                        direction = 'left' if center_x < frame.shape[1] // 2 else 'right'
                        
                        if track_id not in self.counted_vehicles[direction]:
                            self.counted_vehicles[direction].add(track_id)
                            self.vehicle_counts[direction][vehicle_type] += 1
                            
                            # Update current frame statistics
                            stats['by_type'][direction][vehicle_type] += 1
                            stats['total'][direction] += 1
                    
                    # Draw detection
                    self._draw_detection(frame, box, track_id, vehicle_type)
            
            # Draw statistics and timestamp
            self._draw_stats(frame, self.vehicle_counts)
            self._add_timestamp(frame)
            
            return {
                'stats': {
                    'current_frame': stats,
                    'total_counts': self.vehicle_counts
                },
                'frame': frame
            }
            
        except Exception as e:
            print(f"Error in vehicle monitoring: {str(e)}")
            return {
                'stats': {
                    'current_frame': stats,
                    'total_counts': self.vehicle_counts
                },
                'frame': frame
            }
    
    def _draw_detection(self, frame, box, track_id, vehicle_type):
        """Draw detection box with vehicle information"""
        x1, y1, x2, y2 = map(int, box)
        color = self.colors[vehicle_type]
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f'ID: {track_id} {vehicle_type}'
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_stats(self, frame, counts):
        """Draw vehicle statistics on frame"""
        # Left lane stats
        y_pos = 30
        cv2.putText(frame, "Left Lane:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30
        
        for vehicle_type in self.target_classes.values():
            text = f"{vehicle_type}: {counts['left'][vehicle_type]}"
            cv2.putText(frame, text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors[vehicle_type], 2)
            y_pos += 25
        
        # Right lane stats
        y_pos = 30
        x_pos = frame.shape[1] - 200
        cv2.putText(frame, "Right Lane:", (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30
        
        for vehicle_type in self.target_classes.values():
            text = f"{vehicle_type}: {counts['right'][vehicle_type]}"
            cv2.putText(frame, text, (x_pos + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors[vehicle_type], 2)
            y_pos += 25
    
    def _add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)