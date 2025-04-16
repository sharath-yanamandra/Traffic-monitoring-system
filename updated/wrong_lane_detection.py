import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

class WrongLaneDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Colors for visualization
        self.colors = {
            'correct': (0, 255, 0),    # Green
            'wrong': (0, 0, 255)       # Red
        }
        
        # Tracking history
        self.track_history = defaultdict(list)
        self.direction_status = {}
        
        # Number of points to consider for direction
        self.direction_points = 10
        
        # Target vehicle classes (car, motorcycle, bus, truck)
        self.target_classes = [2, 3, 5, 7]
    
    def detect(self, frame):
        """Detect wrong lane driving in a frame"""
        stats = {
            'total_vehicles': 0,
            'wrong_direction': 0,
            'correct_direction': 0
        }
        
        try:
            # Run tracking
            results = self.model.track(frame, persist=True, conf=0.5,
                                     classes=self.target_classes)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Process each detection
                for box, track_id in zip(boxes, track_ids):
                    # Get centroid
                    centroid = self._get_centroid(box)
                    
                    # Add to track history
                    self.track_history[track_id].append(centroid)
                    
                    # Determine direction if enough points
                    direction = self._determine_direction(self.track_history[track_id])
                    if direction:
                        self.direction_status[track_id] = direction
                        stats['total_vehicles'] += 1
                        
                        if direction == 'wrong':
                            stats['wrong_direction'] += 1
                        else:
                            stats['correct_direction'] += 1
                    
                    # Draw detection
                    color = self.colors['wrong'] if self.direction_status.get(track_id) == 'wrong' else self.colors['correct']
                    self._draw_detection(frame, box, track_id, self.direction_status.get(track_id, 'unknown'))
                    
                    # Draw tracking line
                    self._draw_tracking_line(frame, self.track_history[track_id], color)
                
                # Clean up old tracks
                track_ids_set = set(track_ids)
                for track_id in list(self.track_history.keys()):
                    if track_id not in track_ids_set:
                        del self.track_history[track_id]
                        if track_id in self.direction_status:
                            del self.direction_status[track_id]
            
            # Draw statistics and timestamp
            self._draw_stats(frame, stats)
            self._add_timestamp(frame)
            
            return {
                'stats': stats,
                'frame': frame
            }
            
        except Exception as e:
            print(f"Error in wrong lane detection: {str(e)}")
            return {
                'stats': stats,
                'frame': frame
            }
    
    def _get_centroid(self, box):
        """Calculate centroid from bbox coordinates"""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def _determine_direction(self, points):
        """Determine if vehicle is going in correct direction"""
        if len(points) < self.direction_points:
            return None
            
        # Take last n points
        recent_points = points[-self.direction_points:]
        
        # Calculate overall y-direction
        start_y = recent_points[0][1]
        end_y = recent_points[-1][1]
        
        # If y is increasing (going down), it's wrong direction
        return 'wrong' if end_y > start_y else 'correct'
    
    def _draw_detection(self, frame, box, track_id, direction):
        """Draw detection box with direction information"""
        x1, y1, x2, y2 = map(int, box)
        color = self.colors['wrong'] if direction == 'wrong' else self.colors['correct']
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"ID: {track_id} ({direction})"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_tracking_line(self, frame, points, color):
        """Draw tracking line for vehicle"""
        for i in range(1, len(points)):
            cv2.line(frame,
                    points[i - 1],
                    points[i],
                    color, 2)
    
    def _draw_stats(self, frame, stats):
        """Draw statistics on frame"""
        y_pos = 30
        texts = [
            f"Total Vehicles: {stats['total_vehicles']}",
            f"Wrong Direction: {stats['wrong_direction']}",
            f"Correct Direction: {stats['correct_direction']}"
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