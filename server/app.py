from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import numpy as np
import logging
import time
import os
import sys
from io import BytesIO
from PIL import Image

# Add the secret-sauce directory to the Python path so we can import from it
secret_sauce_path = os.path.join(os.path.dirname(__file__), 'secret-sauce')
if secret_sauce_path not in sys.path:
    sys.path.append(secret_sauce_path)

# Try importing OpenCV with error handling for missing dependencies
try:
    import cv2
except ImportError as e:
    print(f"Warning: OpenCV import error: {e}")
    print("You may need to install a different OpenCV variant like 'opencv-contrib-python-headless'")
    cv2 = None

# Import the ASL recognition components from secret-sauce
try:
    from asl_recognition import ASLRecognizer
    from backend import handDetector
except ImportError as e:
    print(f"Error importing ASL recognition components: {e}")
    print("Make sure the secret-sauce directory is properly set up")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Frame processing statistics
stats = {
    'frames_received': 0,
    'frames_processed': 0,
    'start_time': time.time(),
    'processing_times': []
}

# Actual sign language detection model using the secret-sauce
class SignLanguageModel:
    def __init__(self):
        logger.info("Initializing sign language detection model...")
        # Check if OpenCV is available
        self.ready = cv2 is not None
        if not self.ready:
            logger.warning("OpenCV is not available. Using fallback mode.")
            return
        
        try:
            # Set the correct model path - use the one in secret-sauce/models
            model_path = os.path.join(secret_sauce_path, 'models', 'asl_model.h5')
            print("MODEL PATH: ", model_path)
            
            # Initialize the hand detector with ASL recognition
            self.detector = handDetector(detectionCon=0.5, use_asl=True)
            
            # If the detector wasn't able to initialize the ASL recognizer on its own,
            # we'll explicitly initialize it
            if not self.detector.asl_recognizer:
                self.detector.asl_recognizer = ASLRecognizer(model_path)
                self.detector.use_asl = True
            
            # Define custom rules for conflicting predictions from main.py
            self.custom_rules = {
                ("H", "S"): "A",
                ("U", "B"): "B",
                ("C", "Y"): "C",
                ("O", "C"): "C",
                ("R", "D"): "D",
                ("B", "F"): "F",
                ("U", "F"): "F",
                ("X", "I"): "I",
                ("X", "Y"): "I",
                ("R", "I"): "I",
                ("X", "L"): "L",
                ("M", "S"): "M",
                ("M", "X"): "M",
                ("N", "M"): "N",
                ("N", "G"): "N",
                ("S", "T"): "T",
                ("H", "T"): "T",
                ("U", "K"): "U",
                ("V", "K"): "V",
                ("H", "C"): "X",
                ("G", "C"): "X",
                ("P", "M"): "P",
                ("G", "S"): "P",
                ("G", "M"): "P",
                ("Q", "M"): "Q",
            }
            
            # Also ensure class names are loaded
            class_names_path = os.path.join(secret_sauce_path, 'models', 'class_names.txt')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.detector.asl_recognizer.class_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.detector.asl_recognizer.class_names)} class names")
            
            logger.info("ASL recognition model initialized successfully")
            self.ready = True
        except Exception as e:
            logger.error(f"Failed to initialize ASL recognition model: {e}")
            self.ready = False
        
    def predict(self, frame):
        """
        Process a frame to detect and recognize ASL signs
        Returns a dictionary with prediction results
        """
        if not self.ready:
            logger.warning("Model not ready")
            return {"letter": None, "confidence": 0, "error": "Model not ready"}
        
        try:
            # Process the frame with hand detection
            img = self.detector.findHands(frame.copy())
            lmList, bbox = self.detector.findPosition(img, draw=False)
            
            # Initialize result
            result = {
                "letter": None,
                "confidence": 0,
                "alternatives": []
            }
            
            # If a hand is detected and the detector has processed ASL recognition
            if lmList and self.detector.asl_recognizer:
                # Get the model's best prediction
                model_letter, model_confidence = self.detector.get_asl_best()
                
                # Get top-3 predictions for alternatives
                top3 = self.detector.get_asl_top3()
                alternatives = [{"letter": letter, "confidence": float(conf)} for letter, conf in top3]
                
                # Get geometry-based prediction using the main.py implementation
                geometry_letter = self._get_geometry_prediction(lmList)
                
                # Determine final letter using combined approach
                final_letter = self._determine_final_letter(model_letter, model_confidence, geometry_letter)
                
                # Only return a prediction if we're confident enough
                if model_confidence > 0.3 or final_letter:
                    result["letter"] = final_letter or model_letter
                    result["confidence"] = float(model_confidence)
                    result["alternatives"] = alternatives
                    result["geometry_letter"] = geometry_letter
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {"letter": None, "confidence": 0, "error": str(e)}
    
    def _get_geometry_prediction(self, lmList):
        """Geometry-based prediction using hand landmarks based on exact main.py implementation"""
        try:
            # If no landmarks are detected, return None
            if not lmList or len(lmList) < 21:
                return None
                
            # Initialize result
            result = ""
            
            # Define finger parts indices as in main.py
            finger_mcp = [5, 9, 13, 17]
            finger_dip = [6, 10, 14, 18]
            finger_pip = [7, 11, 15, 19]
            finger_tip = [8, 12, 16, 20]
            
            # Initialize fingers list (0.0, 0.25, 0.5, 1.0 values)
            fingers = []
            
            # Calculate finger positions exactly as in main.py
            for id in range(4):
                if(lmList[finger_tip[id]][1]+ 25 < lmList[finger_dip[id]][1] and lmList[16][2]<lmList[20][2]):
                    fingers.append(0.25)
                elif(lmList[finger_tip[id]][2] > lmList[finger_dip[id]][2]):
                    fingers.append(0)
                elif(lmList[finger_tip[id]][2] < lmList[finger_pip[id]][2]): 
                    fingers.append(1)
                elif(lmList[finger_tip[id]][1] > lmList[finger_pip[id]][1] and lmList[finger_tip[id]][1] > lmList[finger_dip[id]][1]): 
                    fingers.append(0.5)
            
            # Check for each letter pattern using EXACT conditions from main.py
            if(lmList[3][2] > lmList[4][2]) and (lmList[3][1] > lmList[6][1])and (lmList[4][2] < lmList[6][2]) and fingers.count(0) == 4:
                result = "A"
                
            elif(lmList[3][1] > lmList[4][1]) and fingers.count(1) == 4:
                result = "B"
            
            elif(lmList[3][1] > lmList[6][1]) and fingers.count(0.5) >= 1 and (lmList[4][2]> lmList[8][2]):
                result = "C"
                
            elif(fingers[0]==1) and fingers.count(0) == 3 and (lmList[3][1] > lmList[4][1]):
                result = "D"
            
            elif (lmList[3][1] < lmList[6][1]) and fingers.count(0) == 4 and lmList[12][2]<lmList[4][2]:
                result = "E"

            elif (fingers.count(1) == 3) and (fingers[0]==0) and (lmList[3][2] > lmList[4][2]):
                result = "F"

            elif(fingers[0]==0.25) and fingers.count(0) == 3:
                result = "G"

            elif(fingers[0]==0.25) and(fingers[1]==0.25) and fingers.count(0) == 2:
                result = "H"
            
            elif (lmList[4][1] < lmList[6][1]) and fingers.count(0) == 3:
                if (len(fingers)==4 and fingers[3] == 1):
                    result = "I"
            
            elif (lmList[4][1] < lmList[6][1] and lmList[4][1] > lmList[10][1] and fingers.count(1) == 2):
                result = "K"
                
            elif(fingers[0]==1) and fingers.count(0) == 3 and (lmList[3][1] < lmList[4][1]):
                result = "L"
            
            elif (lmList[4][1] < lmList[16][1]) and fingers.count(0) == 4:
                result = "M"
            
            elif (lmList[4][1] < lmList[12][1]) and fingers.count(0) == 4:
                result = "N"
                
            elif (lmList[4][1] > lmList[12][1]) and lmList[4][2]<lmList[6][2] and fingers.count(0) == 4:
                result = "T"

            elif (lmList[4][1] > lmList[12][1]) and lmList[4][2]<lmList[12][2] and fingers.count(0) == 4:
                result = "S"
                
            elif(lmList[4][2] < lmList[8][2]) and (lmList[4][2] < lmList[12][2]) and (lmList[4][2] < lmList[16][2]) and (lmList[4][2] < lmList[20][2]):
                result = "O"
            
            elif(fingers[2] == 0) and (lmList[4][2] < lmList[12][2]) and (lmList[4][2] > lmList[6][2]):
                if (len(fingers)==4 and fingers[3] == 0):
                    result = "P"
            
            elif(fingers[1] == 0) and (fingers[2] == 0) and (fingers[3] == 0) and (lmList[8][2] > lmList[5][2]) and (lmList[4][2] < lmList[1][2]):
                result = "Q"
                
            elif(lmList[8][1] < lmList[12][1]) and (fingers.count(1) == 2) and (lmList[9][1] > lmList[4][1]):
                result = "R"
                
            elif (lmList[4][1] < lmList[6][1] and lmList[4][1] < lmList[10][1] and fingers.count(1) == 2 and lmList[3][2] > lmList[4][2] and (lmList[8][1] - lmList[11][1]) <= 50):
                result = "U"
                
            elif (lmList[4][1] < lmList[6][1] and lmList[4][1] < lmList[10][1] and fingers.count(1) == 2 and lmList[3][2] > lmList[4][2]):
                result = "V"
            
            elif (lmList[4][1] < lmList[6][1] and lmList[4][1] < lmList[10][1] and fingers.count(1) == 3):
                result = "W"
            
            elif (fingers[0] == 0.5 and fingers.count(0) == 3 and lmList[4][1] > lmList[6][1]):
                result = "X"
            
            elif(fingers.count(0) == 3) and (lmList[3][1] < lmList[4][1]):
                if (len(fingers)==4 and fingers[3] == 1):
                    result = "Y"
            
            return result if result else None
            
        except Exception as e:
            logger.error(f"Error in geometry prediction: {e}")
            return None
    
    def _determine_final_letter(self, model_letter, model_confidence, geometry_letter):
        """Determine the final letter by combining model and geometry predictions"""
        # If there's no geometry prediction, use the model
        if not geometry_letter:
            return model_letter
        
        # If model and geometry agree, use that letter
        if model_letter == geometry_letter:
            return model_letter
        
        # Check custom rules for known conflicts
        rule_key = (model_letter, geometry_letter)
        if rule_key in self.custom_rules:
            return self.custom_rules[rule_key]
        
        # Default to model's prediction if confidence is high enough
        if model_confidence > 0.7:
            return model_letter
        
        # Fall back to geometry for specific letters that the model struggles with
        geometry_reliable_letters = ["A", "B", "C", "D", "Y"]
        if geometry_letter in geometry_reliable_letters:
            return geometry_letter
        
        # If still undecided, use model prediction
        return model_letter

# Initialize the model
model = SignLanguageModel()

@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'message': 'Sign Language Detection Server',
        'stats': {
            'uptime': time.time() - stats['start_time'],
            'frames_received': stats['frames_received'],
            'frames_processed': stats['frames_processed'],
            'avg_processing_time': sum(stats['processing_times'][-100:]) / max(1, len(stats['processing_times'][-100:])) if stats['processing_times'] else 0
        }
    })

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'status': 'connected', 'message': 'Connection established'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('frame')
def handle_frame(data):
    # Update stats
    stats['frames_received'] += 1
    
    try:
        # Start timing
        start_time = time.time()
        
        # Get the base64 image data and convert to numpy array
        image_data = data.get('image', '')
        
        if not image_data:
            logger.warning("Received empty frame")
            emit('error', {'message': 'Empty frame received'})
            return
            
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            
            # Convert to image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to OpenCV format if available
            if cv2 is not None:
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                # Fallback if OpenCV not available
                frame = np.array(image)
                
            # Run prediction
            if model.ready:
                prediction = model.predict(frame)
                
                # Send prediction back to client
                emit('prediction', {
                    'type': 'prediction',
                    'prediction': prediction,
                    'timestamp': data.get('timestamp', time.time() * 1000)
                })
                
                # Update stats
                stats['frames_processed'] += 1
                processing_time = time.time() - start_time
                stats['processing_times'].append(processing_time)
                
                # Keep only the last 100 processing times
                if len(stats['processing_times']) > 100:
                    stats['processing_times'] = stats['processing_times'][-100:]
                    
                # Log occasionally
                if stats['frames_processed'] % 50 == 0:
                    avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
                    logger.info(f"Processed {stats['frames_processed']} frames. Avg time: {avg_time*1000:.2f}ms")
                    
            else:
                emit('error', {'message': 'Model not ready'})
                
        except Exception as e:
            logger.error(f"Error processing image data: {str(e)}")
            emit('error', {'message': f'Error processing image data: {str(e)}'})
            
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        emit('error', {'message': f'Error processing frame: {str(e)}'})

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5002))
    
    logger.info(f"Starting server on port {port}")
    
    # Start the SocketIO app
    socketio.run(app, host='0.0.0.0', port=port, debug=True)