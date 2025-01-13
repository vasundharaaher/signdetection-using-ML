from flask import Flask, render_template, Response, jsonify
import logging
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_dict = pickle.load(open('./new-model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

# Flag to control prediction
is_predicting = False

# Video streaming generator
def generate_frames():
    global is_predicting, predicted_character
    cap = cv2.VideoCapture(0)
    predicted_character = "None"  # Initialize with a default value
    while True:
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and is_predicting:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract landmarks
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

                # Predict
                prediction = model.predict([np.asarray(data_aux)])
                predicted_label = int(prediction[0])  # Ensure label is an integer
                predicted_character = labels_dict.get(predicted_label, "Unknown")

                # Draw bounding box (without label text)
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Display prediction at the bottom of the frame
        cv2.putText(frame, f"Prediction: {predicted_character}", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame for web
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


# Setup logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/start_predicting', methods=['POST'])
def start_predicting():
    global is_predicting
    is_predicting = True
    app.logger.debug("Prediction started.")
    return jsonify({"status": "Prediction started"})

@app.route('/stop_predicting', methods=['POST'])
def stop_predicting():
    global is_predicting, predicted_character
    is_predicting = False
    predicted_character = "None"  # Reset to default value
    app.logger.debug("Prediction stopped.")
    return jsonify({"status": "Prediction stopped"})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
