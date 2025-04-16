import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

# Load the YOLOv8 model
model = YOLO('path/to/your/yolov8x.pt')  # Replace with your model path

# Define colors for classes
colors = {
    'person': (0, 255, 0),  # Green for person
    'animal': (255, 0, 0)   # Blue for animals
}

# Class IDs for COCO dataset
person_class_id = 0  # Person class
animal_class_ids = [16, 17, 18, 19, 20, 21, 22]  # Bird, cat, dog, horse, sheep, cow, elephant

def process_rtsp_stream(rtsp_url, output_path=None):
    """
    Process RTSP stream for person and animal detection
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
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
                
                # Perform inference on the frame
                results = model(frame)
                
                person_count = 0
                animal_count = 0
                
                for result in results:
                    for box in result.boxes.data:
                        x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
                        conf = float(box[4])                # Confidence score
                        cls = int(box[5])                   # Class ID
                        
                        # Check for 'person' class
                        if cls == person_class_id:
                            person_count += 1
                            label = f'Person {conf:.2f}'
                            color = colors['person']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Check for 'animal' classes
                        elif cls in animal_class_ids:
                            animal_count += 1
                            label = f'Animal {conf:.2f}'
                            color = colors['animal']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display the counts on the frame
                cv2.putText(frame, f'Persons Detected: {person_count}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Animals Detected: {animal_count}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Write frame if output path is provided
                if out is not None:
                    out.write(frame)
                
                # Display frame
                cv2.imshow('Person and Animal Detection', frame)
                
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
    # RTSP stream URL - replace with your RTSP stream URL
    rtsp_url = "rtsp://username:password@ip_address:port/stream"
    
    # Optional: Path to save the processed video
    output_path = f"detection_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        process_rtsp_stream(rtsp_url, output_path)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()