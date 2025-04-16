import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

class SpeedDetector:
    def __init__(self, model_path, real_world_distance=20, speed_limit=40):
        self.model = YOLO(model_path)
        self.real_world_distance = real_world_distance  # in meters
        self.speed_limit = speed_limit  # in km/h
        
        # Tracking data
        self.previous_positions = {}
        self.previous_times = {}
        self.speed_measurements = defaultdict(list)
        
        # Colors for visualization
        self.colors = {
            'normal': (0, 255, 0),    # Green
            'speeding': (0, 0, 255)   # Red
        }
        
        # Region for speed calculation (can be set)
        self.region = None
    
    def set_region(self, points):
        """Set region for speed calculation"""
        self.region = points
    
    def detect(self, frame):
        """Detect vehicle speeds in a frame"""
        if self.region is None:
            self.region = [(0, frame.shape[0]//2), (frame.shape[1], frame.shape[0]//2)]
        
        stats = {
            'speed_violations': 0,
            'total_vehicles': 0,
            'speeds': {}
        }
        
        current_time = datetime.now().timestamp()
        
        try:
            # Run detection and tracking
            results = self.model.track(frame, persist=True, conf=0.5)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Draw speed measurement region
                cv2.line(frame, 
                        (self.region[0][0], self.region[0][1]),
                        (self.region[1][0], self.region[1][1]),
                        (255, 255, 255), 2)
                
                # Process each detection
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Calculate speed
                    speed = self._calculate_speed(track_id, center, current_time)
                    
                    if speed > 0:
                        stats['total_vehicles'] += 1
                        stats['speeds'][track_id] = speed
                        
                        if speed > self.speed_limit:
                            stats['speed_violations'] += 1
                        
                        # Draw detection
                        color = self.colors['speeding'] if speed > self.speed_limit else self.colors['normal']
                        self._draw_detection(frame, box, track_id, speed, color)
            
            # Draw statistics and timestamp
            self._draw_stats(frame, stats)
            self._add_timestamp(frame)
            
            return {
                'stats': stats,
                'frame': frame
            }
            
        except Exception as e:
            print(f"Error in speed detection: {str(e)}")
            return {
                'stats': stats,
                'frame': frame
            }
    
    def _calculate_speed(self, track_id, current_position, current_time):
        """Calculate vehicle speed"""
        if track_id not in self.previous_positions:
            self.previous_positions[track_id] = current_position
            self.previous_times[track_id] = current_time
            return 0
        
        # Calculate time difference
        time_diff = current_time - self.previous_times[track_id]
        if time_diff <= 0:
            return 0
        
        # Calculate pixel distance
        prev_pos = self.previous_positions[track_id]
        pixel_distance = np.sqrt(
            (current_position[0] - prev_pos[0]) ** 2 +
            (current_position[1] - prev_pos[1]) ** 2
        )
        
        # Convert to real-world distance
        region_pixel_distance = np.sqrt(
            (self.region[1][0] - self.region[0][0]) ** 2 +
            (self.region[1][1] - self.region[0][1]) ** 2
        )
        meters_per_pixel = self.real_world_distance / region_pixel_distance
        real_distance = pixel_distance * meters_per_pixel
        
        # Calculate speed in km/h
        speed = (real_distance / time_diff) * 3.6
        
        # Update tracking info
        self.previous_positions[track_id] = current_position
        self.previous_times[track_id] = current_time
        
        # Apply smoothing
        self.speed_measurements[track_id].append(speed)
        if len(self.speed_measurements[track_id]) > 5:
            self.speed_measurements[track_id].pop(0)
        
        return np.mean(self.speed_measurements[track_id])
    
    def _draw_detection(self, frame, box, track_id, speed, color):
        """Draw detection box with speed information"""
        x1, y1, x2, y2 = map(int, box)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f'ID:{track_id} {speed:.1f}km/h'
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_stats(self, frame, stats):
        """Draw statistics on frame"""
        text = f'Speed Violations: {stats["speed_violations"]}'
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   self.colors['speeding'], 2)
        
        text = f'Total Vehicles: {stats["total_vehicles"]}'
        cv2.putText(frame, text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (255, 255, 255), 2)
    
    def _add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)