import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import os
from datetime import datetime
import time

class WrongLaneDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Initialize YOLOv8 model
        self.model = YOLO(model_path)
        
        # Dictionary to store tracking history
        self.track_history = defaultdict(lambda: [])
        
        # Dictionary to store direction status
        self.direction_status = {}  # Will store 'correct' or 'wrong'
        
        # Number of points to consider for direction
        self.direction_points = 10
        
        # RTSP stream settings
        self.frame_skip = 2  # Process every other frame
        self.max_reconnect_attempts = 5
        
    def get_centroid(self, box):
        """Calculate centroid from bbox coordinates"""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def determine_direction(self, points):
        """Determine if vehicle is going in correct direction (top to bottom)"""
        if len(points) < self.direction_points:
            return None
            
        # Take last n points
        recent_points = points[-self.direction_points:]
        
        # Calculate overall y-direction
        start_y = recent_points[0][1]
        end_y = recent_points[-1][1]
        
        # If y is increasing (going down), it's correct direction
        return 'wrong' if end_y > start_y else 'correct'
    
    def process_rtsp_stream(self, rtsp_url, output_path=None):
        """Process RTSP stream for wrong lane detection"""
        # Initialize video capture with RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        
        # Configure RTSP stream settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        if not cap.isOpened():
            raise ValueError("Error: Could not open RTSP stream. Please check the URL and connection.")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        out = None
        if output_path:
            # Generate timestamp-based filename if not provided
            if output_path is True:
                output_path = f"wrong_lane_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps//2, (width, height))
        
        # Variables for frame skipping and reconnection
        frame_count = 0
        reconnect_attempts = 0
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame from stream")
                        reconnect_attempts += 1
                        if reconnect_attempts > self.max_reconnect_attempts:
                            print("Max reconnection attempts reached. Exiting...")
                            break
                        
                        print(f"Attempting to reconnect... ({reconnect_attempts}/{self.max_reconnect_attempts})")
                        cap.release()
                        time.sleep(2)
                        cap = cv2.VideoCapture(rtsp_url)
                        continue
                    
                    # Reset reconnection counter on successful frame grab
                    reconnect_attempts = 0
                    
                    # Skip frames based on frame_skip value
                    frame_count += 1
                    if frame_count % self.frame_skip != 0:
                        continue
                    
                    # Run YOLOv8 tracking on the frame
                    results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7])
                    
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        
                        # Process each detection
                        for box, track_id in zip(boxes, track_ids):
                            # Get centroid
                            centroid = self.get_centroid(box)
                            
                            # Add centroid to track history
                            self.track_history[track_id].append(centroid)
                            
                            # Determine direction if enough points are collected
                            direction = self.determine_direction(self.track_history[track_id])
                            if direction:
                                self.direction_status[track_id] = direction
                            
                            # Draw bounding box
                            color = (0, 0, 255) if self.direction_status.get(track_id) == 'wrong' else (0, 255, 0)
                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                            
                            # Draw tracking line
                            points = self.track_history[track_id]
                            for i in range(1, len(points)):
                                cv2.line(frame, points[i - 1], points[i], color, 2)
                            
                            # Add label with timestamp
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            label = f"ID: {track_id} ({self.direction_status.get(track_id, 'unknown')})"
                            cv2.putText(frame, label, (int(box[0]), int(box[1] - 10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Add timestamp to frame
                            cv2.putText(frame, current_time, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Clean up old tracks
                    if results[0].boxes.id is not None:
                        track_ids_set = set(track_ids)
                        for track_id in list(self.track_history.keys()):
                            if track_id not in track_ids_set:
                                del self.track_history[track_id]
                                if track_id in self.direction_status:
                                    del self.direction_status[track_id]
                    
                    # Write frame if output is enabled
                    if out is not None:
                        out.write(frame)
                    
                    # Display frame
                    cv2.imshow('Wrong Lane Detection', frame)
                    
                    # Break loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue
                    
        finally:
            # Release resources
            print("Cleaning up resources...")
            cap.release()
            if out is not None:
                out.write(frame)
            cv2.destroyAllWindows()

def main():
    try:
        # Initialize detector
        detector = WrongLaneDetector(model_path="path/to/your/yolov8n.pt")
        
        # RTSP stream URL - replace with your RTSP URL
        rtsp_url = "rtsp://username:password@ip_address:port/stream"
        
        # Optional: Path to save the processed video
        output_path = f"wrong_lane_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Process RTSP stream
        detector.process_rtsp_stream(rtsp_url, output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()