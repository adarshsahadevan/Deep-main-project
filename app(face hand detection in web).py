# Q1.Implement real-time face and hand detection and integrate it with a Flask web application for online streaming?
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)

def detect_faces_and_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
    face_results = face_detection.process(rgb_frame)
  
    hand_results = hands.process(rgb_frame)
    
    # drawing face bounding boxes
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # drawing hand landmarks
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            for landmark in landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame

# video frontend
def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_faces_and_hands(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
