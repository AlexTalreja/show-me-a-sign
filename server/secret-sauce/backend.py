import cv2
import time
import os
import mediapipe as mp
import math
import numpy as np

from asl_recognition import ASLRecognizer

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, use_asl=True):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Landmark indices for fingertips
        self.tipIds = [4, 8, 12, 16, 20]
        
        # ASL Recognition
        self.use_asl = use_asl
        self.asl_recognizer = None
        
        self.asl_top3 = []
        
        self.asl_letter = None
        self.asl_confidence = 0.0
        
        if use_asl:
            self._init_asl_recognizer()
    
    def _init_asl_recognizer(self):
        """Initialize the ASL recognizer with a pre-trained model if available."""
        model_dir = "models"
        model_path = os.path.join(model_dir, "asl_model.h5")
        
        if os.path.exists(model_path):
            self.asl_recognizer = ASLRecognizer(model_path)
            print("ASL recognition model loaded successfully.")
        else:
            print("ASL model not found at", model_path)
            print("Please train or provide a model before using ASL recognition.")
            self.use_asl = False

    def findHands(self, img, draw=True):
        """Find hands and optionally draw landmarks."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Find landmark positions of the specified hand."""
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        
        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
            except IndexError:
                return self.lmList, bbox
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax, ymax)

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
                
            # If ASL recognition is enabled, generate wireframe and run the model
            if self.use_asl and self.asl_recognizer:
                self._recognize_asl_gesture_wireframe(myHand, img, bbox)
                
        return self.lmList, bbox

    def _extract_wireframe(self, handLms, img_size=256):
        """
        Create a black canvas of size (img_size x img_size).
        Draw the 21 landmarks in white, connecting them via Mediapipe's HAND_CONNECTIONS.
        """
        wireframe = np.zeros((img_size, img_size), dtype=np.uint8)
        points = []
        
        for lm in handLms.landmark:
            x = int(lm.x * (img_size - 1))
            y = int(lm.y * (img_size - 1))
            points.append((x, y))
        
        # Connections
        for connection in self.mpHands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            x1, y1 = points[start_idx]
            x2, y2 = points[end_idx]
            cv2.line(wireframe, (x1, y1), (x2, y2), (255), 2)
        
        # Points
        for (x, y) in points:
            cv2.circle(wireframe, (x, y), 4, (255), cv2.FILLED)
        
        return wireframe
    
    def _recognize_asl_gesture_wireframe(self, handLms, img, bbox):
        """
        Generate a wireframe image for the hand,
        then get the top 3 ASL predictions and store them internally (self.asl_top3).
        """
        try:
            wireframe_img = self._extract_wireframe(handLms, img_size=256)
            wireframe_resized = cv2.resize(wireframe_img, (64, 64), interpolation=cv2.INTER_AREA)
            self.asl_top3 = self.asl_recognizer.predict_top_k(wireframe_resized, k=3)
            
            # Get the first of the top 3
            self.asl_letter, self.asl_confidence = self.asl_top3[0]
            
            # Display predictions above the bounding box
            for i, (letter, conf) in enumerate(self.asl_top3):
                y_offset = 30 * i
                cv2.putText(
                    img, f"{letter}: {conf:.2f}",
                    (bbox[0] - 20, bbox[1] - 30 - y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )
        
        except Exception as e:
            print(f"ASL wireframe recognition error: {e}")

    def get_asl_top3(self):
        """Return the last computed top-3 predictions from the model."""
        return self.asl_top3
    
    def get_asl_best(self):
        """Return the best guess from the model (letter, confidence)."""
        return self.asl_letter, self.asl_confidence
    
    # ----------------------------------------------------------------
    # GEOMETRY ALGORITHM (kinda just hardcoded lol - ideally wont need this if have more time to train better model):
    # ----------------------------------------------------------------
    def fingersUp(self):
        """Count fingers (example approach)."""
        fingers = []
        if not self.lmList:
            return fingers
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other four
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        """Example function for distance between two landmarks."""
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def backend():
    # Init Camera
    hCam, wCam = 480, 640
    # 0 for front facing webcam, 1 for external webcam
    cap = cv2.VideoCapture(0) # KEEP ON 0 for macbook demo!
    cap.set(4, hCam)
    cap.set(3, wCam)

    # Init detector
    detector = handDetector(detectionCon=0.5, use_asl=True)
    debug = True
    

    pTime = 0
    # Custom combination rules dictionary for when model and geometry disagree lol
    # Format: (model_letter, geometry_letter): confirmed_letter
    custom_rules = {
        ("H", "S"): "A",  # When model predicts F but geometry predicts H, confirm as F
        ("H", "S"): "A",
        ("C", "Y"): "C",
        ("R", "D"): "D",
        ("H", "S"): "A",
        ("B", "F"): "F",
        ("U", "F"): "F",
        ("X", "I"): "I",
        ("X", "Y"): "I",
        ("M", "S"): "M",
        ("M", "X"): "M",
        ("N", "M"): "N",
        ("N", "G"): "N",
        ("S", "T"): "T",
        ("H", "T"): "T",
        ("N", "M"): "N",
        ("U", "K"): "U",
        ("N", "M"): "N",
        ("V", "K"): "V",
        ("H", "C"): "X",
        ("G", "C"): "X",
        ("P", "M"): "P",
        ("G", "S"): "P",
        ("G", "M"): "P",
        ("Q", "M"): "Q",

    }
    
    while True:
        success, img = cap.read()
        if not success:
            print("Camera not found or error reading frame.")
            break
        
        # 1) Detect hands & landmarks
        img = detector.findHands(img)  # draws the hand skeleton
        lmList, bbox = detector.findPosition(img, draw=False)

        # 2) If we have a hand, run geometry-based detection
        geometry_letter = ""
        if len(lmList) != 0:
            try:
                finger_mcp = [5, 9, 13, 17]
                finger_dip = [6, 10, 14, 18]
                finger_pip = [7, 11, 15, 19]
                finger_tip = [8, 12, 16, 20]
                
                fingers = []
                for i in range(4):
                    if (lmList[finger_tip[i]][1] + 25 < lmList[finger_dip[i]][1]
                        and lmList[16][2] < lmList[20][2]):
                        fingers.append(0.25)
                    elif (lmList[finger_tip[i]][2] > lmList[finger_dip[i]][2]):
                        fingers.append(0)
                    elif (lmList[finger_tip[i]][2] < lmList[finger_pip[i]][2]):
                        fingers.append(1)
                    elif (lmList[finger_tip[i]][1] > lmList[finger_pip[i]][1]
                          and lmList[finger_tip[i]][1] > lmList[finger_dip[i]][1]):
                        fingers.append(0.5)
                
                if (lmList[3][2] > lmList[4][2]) and (lmList[3][1] > lmList[6][1]) and (lmList[4][2] < lmList[6][2]) and fingers.count(0) == 4:
                    geometry_letter = "A"
                elif (lmList[3][1] > lmList[4][1]) and fingers.count(1) == 4:
                    geometry_letter = "B"
                elif(lmList[3][1] > lmList[6][1]) and fingers.count(0.5) >= 1 and (lmList[4][2]> lmList[8][2]):
                    geometry_letter = "C"
                    
                elif(fingers[0]==1) and fingers.count(0) == 3 and (lmList[3][1] > lmList[4][1]):
                    geometry_letter = "D"
                
                elif (lmList[3][1] < lmList[6][1]) and fingers.count(0) == 4 and lmList[12][2]<lmList[4][2]:
                    geometry_letter = "E"

                elif (fingers.count(1) == 3) and (fingers[0]==0) and (lmList[3][2] > lmList[4][2]):
                    geometry_letter = "F"

                elif(fingers[0]==0.25) and fingers.count(0) == 3:
                    geometry_letter = "G"

                elif(fingers[0]==0.25) and(fingers[1]==0.25) and fingers.count(0) == 2:
                    geometry_letter = "H"
                
                elif (lmList[4][1] < lmList[6][1]) and fingers.count(0) == 3:
                    if (len(fingers)==4 and fingers[3] == 1):
                        geometry_letter = "I"
                
                elif (lmList[4][1] < lmList[6][1] and lmList[4][1] > lmList[10][1] and fingers.count(1) == 2):
                    geometry_letter = "K"
                    
                elif(fingers[0]==1) and fingers.count(0) == 3 and (lmList[3][1] < lmList[4][1]):
                    geometry_letter = "L"
                
                elif (lmList[4][1] < lmList[16][1]) and fingers.count(0) == 4:
                    geometry_letter = "M"
                
                elif (lmList[4][1] < lmList[12][1]) and fingers.count(0) == 4:
                    geometry_letter = "N"
                    
                elif(lmList[3][1] > lmList[6][1]) and (lmList[3][2] < lmList[6][2]) and fingers.count(0.5) >= 1:
                    geometry_letter = "O"
                
                elif (lmList[4][1] > lmList[12][1]) and lmList[4][2]<lmList[6][2] and fingers.count(0) == 4:
                    geometry_letter = "T"

                elif (lmList[4][1] > lmList[12][1]) and lmList[4][2]<lmList[12][2] and fingers.count(0) == 4:
                    geometry_letter = "S"

                
                elif(lmList[4][2] < lmList[8][2]) and (lmList[4][2] < lmList[12][2]) and (lmList[4][2] < lmList[16][2]) and (lmList[4][2] < lmList[20][2]):
                    geometry_letter = "O"
                
                elif(fingers[2] == 0)  and (lmList[4][2] < lmList[12][2]) and (lmList[4][2] > lmList[6][2]):
                    if (len(fingers)==4 and fingers[3] == 0):
                        geometry_letter = "P"
                
                elif(fingers[1] == 0) and (fingers[2] == 0) and (fingers[3] == 0) and (lmList[8][2] > lmList[5][2]) and (lmList[4][2] < lmList[1][2]):
                    geometry_letter = "Q"
                
                # elif(lmList[10][2] < lmList[8][2] and fingers.count(0) == 4 and lmList[4][2] > lmList[14][2]):
                #     geometry_letter = "Q" 
                    
                elif(lmList[8][1] < lmList[12][1]) and (fingers.count(1) == 2) and (lmList[9][1] > lmList[4][1]):
                    geometry_letter = "R"
                    
                # elif (lmList[3][1] < lmList[6][1]) and fingers.count(0) == 4:
                #     geometry_letter = "S"
                    
                elif (lmList[4][1] < lmList[6][1] and lmList[4][1] < lmList[10][1] and fingers.count(1) == 2 and lmList[3][2] > lmList[4][2] and (lmList[8][1] - lmList[11][1]) <= 50):
                    geometry_letter = "U"
                    
                elif (lmList[4][1] < lmList[6][1] and lmList[4][1] < lmList[10][1] and fingers.count(1) == 2 and lmList[3][2] > lmList[4][2]):
                    geometry_letter = "V"
                
                elif (lmList[4][1] < lmList[6][1] and lmList[4][1] < lmList[10][1] and fingers.count(1) == 3):
                    geometry_letter = "W"
                
                elif (fingers[0] == 0.5 and fingers.count(0) == 3 and lmList[4][1] > lmList[6][1]):
                    geometry_letter = "X"
                
                elif(fingers.count(0) == 3) and (lmList[3][1] < lmList[4][1]):
                    if (len(fingers)==4 and fingers[3] == 1):
                        geometry_letter = "Y"
                
               
                # Only "confirm" geom guess it if it's in top-3 from the model
                top3 = detector.get_asl_top3()  # returns e.g. [("A", 0.95), ("B", 0.03), ...]
                top3_letters = [pair[0] for pair in top3]

                # If geom guess is in the top-3 predictions, confirm
                confirmed_letter = ""
                if geometry_letter and (geometry_letter in top3_letters):
                    confirmed_letter = geometry_letter

                if debug:
                    cv2.putText(img, f"Geometry guess: {geometry_letter}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                model_best_letter, model_best_conf = detector.get_asl_best()
                if debug:
                    if model_best_letter:
                        cv2.putText(img, f"Model best: {model_best_letter} ({model_best_conf:.2f})",
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                confirmed_letter = ""
                
                if not geometry_letter and model_best_letter:
                    # If geometry is empty but model has a prediction, use model
                    confirmed_letter = model_best_letter
                    if debug:
                        cv2.putText(img, "Using model prediction only", 
                                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check if model and geometry agree
                elif model_best_letter == geometry_letter and geometry_letter:
                    confirmed_letter = geometry_letter
                    if debug:
                        cv2.putText(img, "Model and geometry agree", 
                                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # If they disagree, check if we have a custom rule for this combination
                elif model_best_letter and geometry_letter:
                    rule_key = (model_best_letter, geometry_letter)
                    if rule_key in custom_rules:
                        confirmed_letter = custom_rules[rule_key]
                        if debug:
                            cv2.putText(img, f"Custom rule applied", 
                                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        top3 = detector.get_asl_top3()
                        top3_letters = [pair[0] for pair in top3]
                        
                        if geometry_letter in top3_letters:
                            confirmed_letter = geometry_letter
                            if debug:
                                cv2.putText(img, "Geometry in model's top-3", 
                                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                if confirmed_letter:
                    # Display format of confirmed letter
                    cv2.rectangle(img, (28, 355), (128, 455), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, confirmed_letter, (45, 425),
                               cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)

            except IndexError:
                print("Warning: Hand detection incomplete - some landmarks missing")
                # Optionally display a message to the user
                cv2.putText(img, "Position hand fully in frame", (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Show FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime != 0 else 0
        pTime = cTime
        # cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
        #             cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        # Display
        cv2.imshow("ASL Combined Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    backend()
