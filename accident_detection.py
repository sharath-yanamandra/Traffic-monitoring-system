import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import time

class AccidentDetectionSystem:
    def __init__(self, model_path="best.pt", conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the Accident Detection System
        Args:
            model_path: Path to custom trained YOLO model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Define class colors (BGR format)
        self.colors = {
            0: (0, 255, 0),    # bike: green
            1: (0, 0, 255),    # bike_bike_accident: red
            2: (0, 0, 255),    # bike_object_accident: red
            3: (0, 0, 255),    # bike_person_accident: red
            4: (0, 255, 0),    # car: green
            5: (0, 0, 255),    # car_bike_accident: red
            6: (0, 0, 255),    # car_car_accident: red
            7: (0, 0, 255),    # car_object_accident: red
            8: (0, 0, 255),    # car_person_accident: red
            9: (0, 255, 0)     # person: green
        }

        # Initialize class names
        self.class_names = {
            0: 'bike',
            1: 'bike_bike_accident',
            2: 'bike_object_accident',
            3: 'bike_person_accident',
            4: 'car',
            5: 'car_bike_accident',
            6: 'car_car_accident',
            7: 'car_object_accident',
            8: 'car_person_accident',
            9: 'person'
        }

        # Initialize accident event logger
        self.accident_log = []

    def is_accident_class(self, class_id):
        """Check if the class ID represents an accident"""
        accident_classes = [1, 2, 3, 5, 6, 7, 8]  # All accident-related class IDs
        return class_id in accident_classes

    def log_accident(self, class_name, confidence, frame_time):
        """Log accident details with timestamp"""
        accident_entry = {
            'timestamp': frame_time,
            'type': class_name,
            'confidence': confidence
        }
        self.accident_log.append(accident_entry)
        
        # Immediately write to file for real-time logging
        with open("realtime_accident_log.txt", 'a') as f:
            f.write(f"\nTime: {frame_time}\n")
            f.write(f"Type: {class_name}\n")
            f.write(f"Confidence: {confidence:.2f}\n")
            f.write("---------------------\n")

    def process_frame(self, frame):
        """
        Process a single frame for accident detection
        Args:
            frame: Input frame (numpy array)
        Returns:
            frame: Processed frame with annotations
            bool: Whether an accident was detected in this frame
        """
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        
        # Initialize accident detected flag
        accident_detected = False
        
        # Process detections
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get color for this class
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        color, 2)
            
            # Prepare label
            label = f"{self.class_names[class_id]} {confidence:.2f}"
            
            # Add label to frame
            cv2.putText(frame, label, 
                       (int(x1), int(y1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check if this is an accident class
            if self.is_accident_class(class_id):
                accident_detected = True
                self.log_accident(self.class_names[class_id], confidence, current_time)
        
        # Add timestamp to frame
        cv2.putText(frame, current_time, 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # If accident detected, add warning text
        if accident_detected:
            cv2.putText(frame, "ACCIDENT DETECTED!", 
                       (frame.shape[1]//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, accident_detected

    def process_rtsp_stream(self, rtsp_url, output_path=None):
        """
        Process RTSP stream for accident detection
        Args:
            rtsp_url: RTSP stream URL
            output_path: Optional path to save output video
        """
        # Initialize video capture with RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        
        # Configure RTSP stream settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Minimize buffer size
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        if not cap.isOpened():
            raise Exception("Error: Could not open RTSP stream. Please check the URL and network connection.")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Variables for frame skipping and reconnection
        frame_count = 0
        max_reconnect_attempts = 5
        reconnect_attempts = 0
        frame_skip = 2  # Process every other frame
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame from stream")
                        reconnect_attempts += 1
                        if reconnect_attempts > max_reconnect_attempts:
                            print("Max reconnection attempts reached. Exiting...")
                            break
                        
                        print(f"Attempting to reconnect... ({reconnect_attempts}/{max_reconnect_attempts})")
                        cap.release()
                        time.sleep(2)
                        cap = cv2.VideoCapture(rtsp_url)
                        continue
                    
                    # Reset reconnection counter on successful frame grab
                    reconnect_attempts = 0
                    
                    # Skip frames based on frame_skip value
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue
                    
                    # Process frame
                    processed_frame, accident_detected = self.process_frame(frame)
                    
                    # Write frame if output path is provided
                    if out is not None:
                        out.write(processed_frame)
                    
                    # Display frame
                    cv2.imshow('Accident Detection System', processed_frame)
                    
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
                out.release()
            cv2.destroyAllWindows()

def main():
    # Initialize detector
    detector = AccidentDetectionSystem(
        model_path="path/to/your/best.pt",  # Update with your model path
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # RTSP stream URL - replace with your RTSP stream URL
    rtsp_url = "rtsp://username:password@ip_address:port/stream"
    
    # Optional: Path to save the processed video
    output_path = f"accident_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    # Process RTSP stream
    detector.process_rtsp_stream(rtsp_url, output_path)

if __name__ == "__main__":
    main()