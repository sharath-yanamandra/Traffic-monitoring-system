import cv2
import numpy as np
import threading
import queue
import time
import json
import base64
from datetime import datetime

# Import all detectors
from helmet_detection import HelmetDetector
from accident_detection import AccidentDetector
from speed_detection import SpeedDetector
from triple_riding_detection import TripleRidingDetector
from wrong_lane_detection import WrongLaneDetector
from vehicle_monitoring import VehicleMonitor
from people_and_animals import PeopleAndAnimalsDetector

class TrafficMonitoringSystem:
    def __init__(self):
        self.models = {}
        self.stream_queues = {}
        self.processing_threads = {}
        self.active_cameras = set()
        
        # Initialize statistics with thread safety
        self.stats_lock = threading.Lock()
        self.statistics = {
            'helmet_violations': {
                'total': 0,
                'current_hour': 0,
                'with_helmet': 0,
                'without_helmet': 0
            },
            'accidents': {
                'total': 0,
                'current': False,
                'current_count': 0
            },
            'speed_violations': {
                'total': 0,
                'current_violations': 0,
                'average_speed': 0
            },
            'triple_riding': {
                'total': 0,
                'current_violations': 0
            },
            'wrong_lane': {
                'total': 0,
                'current_violations': 0
            },
            'vehicle_count': {
                'left': 0,
                'right': 0
            },
            'people_animals': {
                'people': 0,
                'animals': 0
            }
        }
        
    def initialize_models(self):
        """Initialize all detection models"""
        try:
            # Initialize all detectors with their respective weights
            self.models['helmet'] = HelmetDetector('weights/helmet_weight.pt')
            self.models['accident'] = AccidentDetector('weights/accident_weight.pt')
            self.models['speed'] = SpeedDetector('weights/speed_weight.pt')
            self.models['triple_riding'] = TripleRidingDetector('weights/triple_riding_weight.pt')
            self.models['wrong_lane'] = WrongLaneDetector('weights/wrong_lane_weight.pt')
            self.models['vehicle'] = VehicleMonitor('weights/vehicle_weight.pt')
            self.models['people_animals'] = PeopleAndAnimalsDetector('weights/yolov8x.pt')
            
            # Set regions for specific detectors
            if 'speed' in self.models:
                self.models['speed'].set_region([(100, 300), (500, 300)])
            
            if 'vehicle' in self.models:
                self.models['vehicle'].set_counting_line(350)
            
            print("All models initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            return False

    def process_frame(self, frame, camera_id):
        """Process a single frame through all detection models"""
        try:
            results = {
                'frame_time': datetime.now().isoformat(),
                'camera_id': camera_id,
                'detections': {}
            }
            
            # Process through each model
            if 'helmet' in self.models:
                helmet_results = self.models['helmet'].detect(frame)
                with self.stats_lock:
                    self.statistics['helmet_violations']['with_helmet'] = helmet_results['stats']['With_Helmet']
                    self.statistics['helmet_violations']['without_helmet'] = helmet_results['stats']['Without_Helmet']
                    self.statistics['helmet_violations']['total'] += helmet_results['stats']['Without_Helmet']
                results['detections']['helmet'] = helmet_results['stats']
                frame = helmet_results['frame']
            
            if 'accident' in self.models:
                accident_results = self.models['accident'].detect(frame)
                with self.stats_lock:
                    self.statistics['accidents']['current'] = accident_results['stats']['accident_detected']
                    self.statistics['accidents']['current_count'] = accident_results['stats']['accident_count']
                    if accident_results['stats']['accident_detected']:
                        self.statistics['accidents']['total'] += 1
                results['detections']['accident'] = accident_results['stats']
                frame = accident_results['frame']
            
            if 'speed' in self.models:
                speed_results = self.models['speed'].detect(frame)
                with self.stats_lock:
                    self.statistics['speed_violations']['current_violations'] = speed_results['stats']['speed_violations']
                    self.statistics['speed_violations']['total'] += speed_results['stats']['speed_violations']
                results['detections']['speed'] = speed_results['stats']
                frame = speed_results['frame']
            
            if 'triple_riding' in self.models:
                triple_results = self.models['triple_riding'].detect(frame)
                with self.stats_lock:
                    self.statistics['triple_riding']['current_violations'] = triple_results['stats']['triple_riding']
                    self.statistics['triple_riding']['total'] += triple_results['stats']['triple_riding']
                results['detections']['triple_riding'] = triple_results['stats']
                frame = triple_results['frame']
            
            if 'wrong_lane' in self.models:
                wrong_lane_results = self.models['wrong_lane'].detect(frame)
                with self.stats_lock:
                    self.statistics['wrong_lane']['current_violations'] = wrong_lane_results['stats']['wrong_direction']
                    self.statistics['wrong_lane']['total'] += wrong_lane_results['stats']['wrong_direction']
                results['detections']['wrong_lane'] = wrong_lane_results['stats']
                frame = wrong_lane_results['frame']
            
            if 'vehicle' in self.models:
                vehicle_results = self.models['vehicle'].detect(frame)
                with self.stats_lock:
                    self.statistics['vehicle_count']['left'] = sum(vehicle_results['stats']['current_frame']['total']['left'])
                    self.statistics['vehicle_count']['right'] = sum(vehicle_results['stats']['current_frame']['total']['right'])
                results['detections']['vehicle'] = vehicle_results['stats']
                frame = vehicle_results['frame']
            
            if 'people_animals' in self.models:
                pa_results = self.models['people_animals'].detect(frame)
                with self.stats_lock:
                    self.statistics['people_animals']['people'] = pa_results['stats']['person_count']
                    self.statistics['people_animals']['animals'] = pa_results['stats']['animal_count']
                results['detections']['people_animals'] = pa_results['stats']
                frame = pa_results['frame']
            
            return results, frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None, frame

    def start_stream_processing(self, camera_id, rtsp_url):
        """Start processing an RTSP stream"""
        if camera_id in self.active_cameras:
            return False
        
        try:
            # Create a queue for this camera
            self.stream_queues[camera_id] = queue.Queue(maxsize=10)
            
            # Start processing thread
            thread = threading.Thread(
                target=self._process_stream,
                args=(camera_id, rtsp_url),
                daemon=True
            )
            thread.start()
            
            self.processing_threads[camera_id] = thread
            self.active_cameras.add(camera_id)
            return True
            
        except Exception as e:
            print(f"Error starting stream: {str(e)}")
            return False

    def _process_stream(self, camera_id, rtsp_url):
        """Process RTSP stream in a separate thread"""
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print(f"Failed to open RTSP stream for camera {camera_id}")
            return
            
        skip_frames = 2  # Process every other frame
        frame_counter = 0
        
        try:
            while camera_id in self.active_cameras:
                frame_counter += 1
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Failed to read from camera {camera_id}")
                    time.sleep(1)
                    # Attempt to reconnect
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url)
                    continue

                if frame_counter % skip_frames == 0:
                    # Process frame through all models
                    results, processed_frame = self.process_frame(frame, camera_id)
                    
                    if results:
                        # Convert frame to base64 for sending over WebSocket
                        _, buffer = cv2.imencode('.jpg', processed_frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Prepare WebSocket payload
                        payload = {
                            'frame': frame_base64,
                            'results': results
                        }
                        
                        # Send through WebSocket (assuming socketio is available)
                        try:
                            from api import socketio
                            socketio.emit(f'detection_results_{camera_id}', payload)
                        except Exception as e:
                            print(f"Error sending WebSocket message: {str(e)}")
                            
        except Exception as e:
            print(f"Error in stream processing: {str(e)}")
        finally:
            cap.release()
            self.active_cameras.remove(camera_id)
            
    def stop_stream_processing(self, camera_id):
        """Stop processing a specific camera stream"""
        if camera_id in self.active_cameras:
            self.active_cameras.remove(camera_id)
            return True
        return False
        
    def get_statistics(self):
        """Get current statistics with thread safety"""
        with self.stats_lock:
            return self.statistics.copy()

# Create global instance
traffic_system = TrafficMonitoringSystem()

# Initialize Flask routes if being used with Flask
try:
    from flask import Flask, jsonify, request
    from flask_socketio import SocketIO
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @app.route('/api/start_stream', methods=['POST'])
    def start_stream():
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
    
    @app.route('/api/stop_stream', methods=['POST'])
    def stop_stream():
        data = request.json
        camera_id = data.get('camera_id')
        
        if not camera_id:
            return jsonify({'error': 'Missing camera_id'}), 400
            
        success = traffic_system.stop_stream_processing(camera_id)
        
        if success:
            return jsonify({'message': 'Stream processing stopped'})
        else:
            return jsonify({'error': 'Stream not found'}), 404
    
    @app.route('/api/statistics', methods=['GET'])
    def get_statistics():
        return jsonify(traffic_system.get_statistics())
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
    
    if __name__ == '__main__':
        # Initialize models before starting server
        if traffic_system.initialize_models():
            socketio.run(app, host='0.0.0.0', port=5000)
        else:
            print("Failed to initialize models. Exiting...")

except ImportError:
    print("Flask not installed. Web API not available.")