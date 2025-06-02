import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3  # Import the text-to-speech library
import time  # Import time for controlling frame rate

# Load the trained model
with open('sign_language_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label map (update if you saved labels separately)
with open('labels_dict.pkl', 'rb') as f:
    label_map = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to extract 126 features (63 per hand)
def extract_126_features(results):
    features = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        # Pad if only one hand is detected
        if len(results.multi_hand_landmarks) == 1:
            features.extend([0] * 63)  # 21 × 3
        return features if len(features) == 126 else None
    return None

# Initialize webcam
cap = cv2.VideoCapture(0)
sequence_buffer = []
SEQUENCE_LENGTH = 30

# Timer settings
last_prediction_time = time.time()
prediction_interval = 5 # Interval between predictions in seconds

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip & convert BGR to RGB
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Feature extraction
    features = extract_126_features(results)

    if features:
        sequence_buffer.append(features)
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        # Process prediction only if enough time has passed since the last prediction
        if len(sequence_buffer) == SEQUENCE_LENGTH and (time.time() - last_prediction_time) >= prediction_interval:
            last_prediction_time = time.time()  # Update the last prediction time

            input_sequence = np.array(sequence_buffer).flatten().reshape(1, -1)  # (1, 3780)
            prediction = model.predict(input_sequence)[0]
            predicted_label = label_map.get(prediction, "Unknown")
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Prediction:", predicted_label)
            
            # Speak the prediction aloud
            engine.say(predicted_label)
            engine.runAndWait()
    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
