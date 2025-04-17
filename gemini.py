import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dataset directory
DATASET_DIR = 'handsign_dataset'
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def process_frame(image, hand_sign_label):
    """Processes a single frame to detect hands and store landmark data."""
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    data = []

    if results.multi_handedness and results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label  # 'Left' or 'Right'
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            data.append([hand_sign_label, hand_type] + landmarks)
            # Optional: Draw landmarks on the frame for visualization
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image, data

def record_handsign(hand_sign_label):
    """Records a handsign from the webcam and saves the data."""
    cap = cv2.VideoCapture(0)
    recording_data = []
    print(f"Recording hand sign: '{hand_sign_label}'. Press 'q' to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)

        processed_frame, frame_data = process_frame(frame.copy(), hand_sign_label)
        recording_data.extend(frame_data)

        cv2.imshow('Hand Sign Recorder', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return recording_data

def save_to_dataset(data, hand_sign_label):
    """Saves the recorded data to a CSV file."""
    if not data:
        print(f"No data recorded for '{hand_sign_label}'.")
        return

    filename = os.path.join(DATASET_DIR, f"{hand_sign_label}.csv")
    column_names = ['label', 'hand_type']
    for i in range(21):  # 21 landmarks per hand
        column_names.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])

    df = pd.DataFrame(data, columns=column_names)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"Data appended to '{filename}' for hand sign '{hand_sign_label}'.")
    else:
        df.to_csv(filename, index=False)
        print(f"Data saved to '{filename}' for hand sign '{hand_sign_label}'.")

def main():
    while True:
        hand_sign = input("Enter the name of the hand sign you want to record (or 'exit' to quit): ").strip()
        if hand_sign.lower() == 'exit':
            break
        if not hand_sign:
            print("Hand sign name cannot be empty.")
            continue

        print(f"Ready to record the hand sign: '{hand_sign}'.")
        input("Press Enter when you are ready to start recording...")
        recorded_data = record_handsign(hand_sign)
        save_to_dataset(recorded_data, hand_sign)
        print("\n")

if __name__ == "__main__":
    main()