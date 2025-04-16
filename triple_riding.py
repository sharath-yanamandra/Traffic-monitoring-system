import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict
import time
from datetime import datetime

class TripleRiderDetector:
    def __init__(self, model_path="yolov8x.pt"):
        # Initialize YOLOv8 model
        self.model = YOLO(model_path)
        
        # Constants
        self.MOTORCYCLE_CLASS = 3  # COCO class for motorcycle
        self.PERSON_CLASS = 0      # COCO class for person
        self.CONFIDENCE_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.3
        self.TRACKING_HISTORY = defaultdict(list)
        self.MOTORCYCLE_RIDERS = defaultdict(int)
        self.DETECTION_PERSISTENCE = 5  # frames
        
        # RTSP settings
        self.frame_skip = 2  # Process every other frame
        self.max_reconnect_attempts = 5
        
    def calculate_overlap(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / min(area1, area2)
    
    def expand_box(self, box, frame_shape, expand_ratio=1.3):
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
    
    def process_frame(self, frame):
        """Process a single frame to detect triple riding"""
        # Run YOLOv8 tracking
        results = self.model.track(frame, persist=True, conf=self.CONFIDENCE_THRESHOLD, 
                                 classes=[self.MOTORCYCLE_CLASS, self.PERSON_CLASS])
        
        if results[0].boxes.id is None:
            return frame
            
        # Extract detections
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Separate motorcycles and persons
        motorcycles = []
        motorcycle_ids = []
        persons = []
        
        for box, cls, track_id, conf in zip(boxes, classes, track_ids, confidences):
            if conf < self.CONFIDENCE_THRESHOLD:
                continue
                
            if cls == self.MOTORCYCLE_CLASS:
                motorcycles.append(box)
                motorcycle_ids.append(track_id)
            elif cls == self.PERSON_CLASS:
                persons.append(box)
        
        # Add timestamp to frame
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Process each motorcycle
        for motorcycle_idx, (motorcycle_box, motorcycle_id) in enumerate(zip(motorcycles, motorcycle_ids)):
            expanded_box = self.expand_box(motorcycle_box, frame.shape)
            
            rider_count = 0
            for person_box in persons:
                overlap = self.calculate_overlap(expanded_box, person_box)
                if overlap > self.IOU_THRESHOLD:
                    rider_count += 1
            
            self.TRACKING_HISTORY[motorcycle_id].append(rider_count)
            if len(self.TRACKING_HISTORY[motorcycle_id]) > self.DETECTION_PERSISTENCE:
                self.TRACKING_HISTORY[motorcycle_id].pop(0)
            
            smooth_count = max(1, round(np.mean(self.TRACKING_HISTORY[motorcycle_id])))
            self.MOTORCYCLE_RIDERS[motorcycle_id] = smooth_count
            
            if smooth_count >= 3:
                color = (0, 0, 255)  # Red for triple riding
                text = "TRIPLE RIDING DETECTED!"
            else:
                color = (0, 255, 0)  # Green for normal riding
                text = f"Riders: {smooth_count}"
            
            # Draw annotations
            cv2.rectangle(frame, 
                         (int(motorcycle_box[0]), int(motorcycle_box[1])),
                         (int(motorcycle_box[2]), int(motorcycle_box[3])),
                         color, 2)
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame,
                         (int(motorcycle_box[0]), int(motorcycle_box[1] - text_size[1] - 10)),
                         (int(motorcycle_box[0] + text_size[0]), int(motorcycle_box[1])),
                         color, -1)
            cv2.putText(frame, text,
                       (int(motorcycle_box[0]), int(motorcycle_box[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Clean up old tracks
        for track_id in list(self.TRACKING_HISTORY.keys()):
            if track_id not in motorcycle_ids:
                del self.TRACKING_HISTORY[track_id]
                if track_id in self.MOTORCYCLE_RIDERS:
                    del self.MOTORCYCLE_RIDERS[track_id]
        
        return frame
    
    def process_rtsp_stream(self, rtsp_url, output_path=None):
        """Process RTSP stream for triple rider detection"""
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
                output_path = f"triple_rider_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps//2, (width, height))
        
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
                    
                    reconnect_attempts = 0
                    
                    frame_count += 1
                    if frame_count % self.frame_skip != 0:
                        continue
                    
                    processed_frame = self.process_frame(frame)
                    
                    if out is not None:
                        out.write(processed_frame)
                    
                    cv2.imshow('Triple Rider Detection', cv2.resize(processed_frame, (1280, 720)))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue
                    
        finally:
            print("Cleaning up resources...")
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

def main():
    try:
        # Initialize detector
        detector = TripleRiderDetector(model_path="path/to/your/yolov8x.pt")
        
        # RTSP stream URL - replace with your RTSP URL
        rtsp_url = "rtsp://username:password@ip_address:port/stream"
        
        # Optional: Path to save the processed video
        output_path = f"triple_rider_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Process RTSP stream
        detector.process_rtsp_stream(rtsp_url, output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()