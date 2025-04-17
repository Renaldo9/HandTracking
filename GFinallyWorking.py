import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained model and scaler
MODEL_FILENAME = 'handsign_model.pkl'
try:
    with open(MODEL_FILENAME, 'rb') as file:
        model, scaler = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILENAME}' not found. Please train the model first.")
    exit()

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def predict_handsign(image):
    """Predicts the hand sign shown in the image."""
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    predictions = []

    if results.multi_handedness and results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label  # 'Left' or 'Right'
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Scale the landmarks
            scaled_landmarks = scaler.transform(np.array(landmarks).reshape(1, -1))

            # Make prediction
            predicted_label = model.predict(scaled_landmarks)[0]
            predictions.append((predicted_label, hand_type))

            # Optional: Draw landmarks on the frame
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return image, predictions

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)

        processed_frame, predictions = predict_handsign(frame.copy())

        # Display the predictions on the frame
        y_position = 30
        for prediction, hand_type in predictions:
            text = f"{prediction} ({hand_type})"
            cv2.putText(processed_frame, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            y_position += 30

        cv2.imshow('Real-time Hand Sign Prediction', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()