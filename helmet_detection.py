import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

class HelmetDetector:
    def __init__(self, model_path, rtsp_url, output_path=None):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Store RTSP URL
        self.rtsp_url = rtsp_url
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(rtsp_url)
        
        # Configure RTSP stream settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open RTSP stream. Please check the URL and connection.")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        self.out = None
        if output_path:
            self.out = cv2.VideoWriter(output_path, 
                                     cv2.VideoWriter_fourcc(*'mp4v'), 
                                     self.fps, 
                                     (self.frame_width, self.frame_height))
        
        # Colors for visualization
        self.colors = {
            'With_Helmet': (0, 255, 0),    # Green
            'Without_Helmet': (0, 0, 255)   # Red
        }
        
        # Counters for statistics
        self.stats = {
            'With_Helmet': 0,
            'Without_Helmet': 0
        }
        
        # Frame processing variables
        self.frame_skip = 2  # Process every other frame
        self.max_reconnect_attempts = 5
        
    def draw_detection_box(self, frame, box, cls, conf):
        """Draw detection box with label and confidence"""
        x1, y1, x2, y2 = map(int, box)
        color = self.colors[cls]
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label with class and confidence
        label = f'{cls} {conf:.2f}'
        
        # Get label size
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(frame, 
                     (x1, y1 - label_height - 10), 
                     (x1 + label_width + 10, y1), 
                     color, 
                     -1)
        
        # Draw label text
        cv2.putText(frame, label, 
                    (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 255, 255), 
                    2)

    def draw_stats(self, frame):
        """Draw detection statistics on frame"""
        # Background for stats
        cv2.rectangle(frame, 
                     (10, 10), 
                     (250, 100), 
                     (0, 0, 0), 
                     -1)
        
        # Draw stats text
        y_pos = 40
        for cls, count in self.stats.items():
            color = self.colors[cls]
            text = f'{cls}: {count}'
            cv2.putText(frame, text, 
                       (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2)
            y_pos += 30
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (10, self.frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)

    def process_rtsp_stream(self):
        """Process the RTSP stream and detect helmets"""
        frame_count = 0
        reconnect_attempts = 0
        
        try:
            while True:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame from stream")
                        reconnect_attempts += 1
                        if reconnect_attempts > self.max_reconnect_attempts:
                            print("Max reconnection attempts reached. Exiting...")
                            break
                        
                        print(f"Attempting to reconnect... ({reconnect_attempts}/{self.max_reconnect_attempts})")
                        self.cap.release()
                        time.sleep(2)
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        continue
                    
                    # Reset reconnection counter on successful frame grab
                    reconnect_attempts = 0
                    
                    # Skip frames based on frame_skip value
                    frame_count += 1
                    if frame_count % self.frame_skip != 0:
                        continue
                    
                    # Reset counters for each processed frame
                    self.stats = {
                        'With_Helmet': 0,
                        'Without_Helmet': 0
                    }
                    
                    # Run detection with increased confidence threshold
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
                            self.stats[cls] += 1
                            
                            # Draw detection box
                            self.draw_detection_box(frame, [x1, y1, x2, y2], cls, conf)
                    
                    # Draw statistics
                    self.draw_stats(frame)
                    
                    # Show frame
                    cv2.imshow('Helmet Detection', frame)
                    
                    # Write frame to output video if enabled
                    if self.out is not None:
                        self.out.write(frame)
                    
                    # Break loop on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    continue
                    
        finally:
            # Release resources
            print("Cleaning up resources...")
            self.cap.release()
            if self.out is not None:
                self.out.release()
            cv2.destroyAllWindows()

def main():
    # Configuration
    model_path = 'path/to/your/best1.pt'  # Update with your model path
    rtsp_url = "rtsp://username:password@ip_address:port/stream"  # Update with your RTSP URL
    
    # Optional: Path to save the processed video
    output_path = f"helmet_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        # Create and run detector
        detector = HelmetDetector(model_path, rtsp_url, output_path)
        detector.process_rtsp_stream()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()