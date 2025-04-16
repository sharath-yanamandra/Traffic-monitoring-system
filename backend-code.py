from flask import Flask, jsonify, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import numpy as np
import threading
import queue
import time
import json
from datetime import datetime

from accident_detection import AccidentDetectionSystem
from helmet_detection import HelmetDetector
from speed import main

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class TrafficMonitoringSystem:
    def __init__(self):
        self.models = {}
        self.stream_queues = {}
        self.processing_threads = {}
        self.statistics = {
            'accidents': 0,
            'helmet_violations': 0,
            'triple_riding': 0,
            'wrong_lane': 0,
            'speed_violations': 0
        }
        
    def initialize_models(self):
        """Initialize all detection models"""
        # Initialize your models
        self.models['accident'] =  AccidentDetectionSystem(
            model_path="path/to/your/best.pt",  # Update with your model path
            conf_threshold=0.5,
            iou_threshold=0.45
        )
        self.models['helmet'] = HelmetDetector(model_path, rtsp_url, output_path)
        self.models['people_animals'] = YourPeopleAnimalsModel()
        self.models['speed'] = main()
        self.models['triple_riding'] = YourTripleRidingModel()
        self.models['vehicle'] = YourVehicleModel()
    
    def process_frame(self, frame, camera_id):
        """Process a single frame through all models"""
        results = {
            'frame_time': datetime.now().isoformat(),
            'detections': {}
        }
        
        # Process through each model
        # Accident detection
        if 'accident' in self.models:
            results['detections']['accidents'] = self.models['accident'].detect(frame)
            
        # Helmet detection
        if 'helmet' in self.models:
            results['detections']['helmet'] = self.models['helmet'].detect(frame)
            
        # Add other detections similarly
        
        return results, frame

    def start_stream_processing(self, camera_id, rtsp_url):
        """Start processing an RTSP stream"""
        if camera_id in self.processing_threads:
            return False
        
        # Create a queue for this camera
        self.stream_queues[camera_id] = queue.Queue(maxsize=10)
        
        # Start processing thread
        thread = threading.Thread(
            target=self._process_stream,
            args=(camera_id, rtsp_url)
        )
        thread.daemon = True
        thread.start()
        
        self.processing_threads[camera_id] = thread
        return True

    def _process_stream(self, camera_id, rtsp_url):
        """Process RTSP stream in a separate thread"""
        cap = cv2.VideoCapture(rtsp_url)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read from camera {camera_id}")
                time.sleep(1)
                continue
                
            # Process frame through all models
            results, processed_frame = self.process_frame(frame, camera_id)
            
            # Convert frame to JPEG
            _, jpeg = cv2.imencode('.jpg', processed_frame)
            frame_bytes = jpeg.tobytes()
            
            # Update statistics
            self._update_statistics(results)
            
            # Emit results through Socket.IO
            socketio.emit(f'detection_results_{camera_id}', {
                'frame': frame_bytes,
                'results': results
            })
            
            # Control processing rate
            time.sleep(0.033)  # ~30 FPS
    
    def _update_statistics(self, results):
        """Update violation statistics"""
        if 'accidents' in results['detections']:
            self.statistics['accidents'] += len(results['detections']['accidents'])
        # Update other statistics similarly

# Initialize the system
traffic_system = TrafficMonitoringSystem()
traffic_system.initialize_models()

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    """Start processing a new RTSP stream"""
    data = request.json
    camera_id = data.get('camera_id')
    rtsp_url = data.get('rtsp_url')
    
    if not camera_id or not rtsp_url:
        return jsonify({'error': 'Missing required parameters'}), 400
        
    success = traffic_system.start_stream_processing(camera_id, rtsp_url)
    
    if success:
        return jsonify({'message': 'Stream processing started'})
    else:
        return jsonify({'error': 'Stream already being processed'}), 400

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get current violation statistics"""
    return jsonify(traffic_system.statistics)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
