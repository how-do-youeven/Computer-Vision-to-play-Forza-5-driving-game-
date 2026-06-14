import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import pyautogui

# Load trained model and scaler
model = tf.keras.models.load_model("gesture_mlp_model.h5")
scaler = joblib.load("scaler.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Gesture to key mapping
gesture_key_map = {0: 'w', 1: 'a', 2: 's', 3: 'd', 4: 'space'}  # Map gestures to keys
pressed_keys = {key: False for key in gesture_key_map.values()}

def process_landmarks(hand_landmarks):
    """Extracts and normalizes hand landmark coordinates."""
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark]).flatten()
    return scaler.transform([landmarks])

def press_key(gesture):
    """Handles key press logic."""
    key = gesture_key_map.get(gesture)
    if key:
        if not pressed_keys[key]:
            pyautogui.keyDown(key)
            pressed_keys[key] = True
    
def release_keys():
    """Releases all mapped keys."""
    for key, pressed in pressed_keys.items():
        if pressed:
            pyautogui.keyUp(key)
            pressed_keys[key] = False

def compare_hand_averages(left_hand, right_hand, threshold=0.02):
    """Compares the average landmark values of both hands and presses 'a' or 'd' accordingly."""
    left_avg = np.mean([lm.x + lm.y for lm in left_hand]) if left_hand else None
    right_avg = np.mean([lm.x + lm.y for lm in right_hand]) if right_hand else None
    
    if left_avg is not None and right_avg is not None:
        diff = abs(left_avg - right_avg)
        if diff > threshold:
            if left_avg > right_avg:
                pyautogui.keyDown('a')
                pyautogui.keyUp('d')
            else:
                pyautogui.keyDown('d')
                pyautogui.keyUp('a')
        else:
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')

# Start video capture
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Unable to read from camera. Exiting...")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    release_keys()
    
    left_hand, right_hand = None, None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = process_landmarks(hand_landmarks)
            prediction = model.predict(features)
            gesture = np.argmax(prediction)
            press_key(gesture)
            
            if handedness.classification[0].label == "Left":
                left_hand = hand_landmarks.landmark
            else:
                right_hand = hand_landmarks.landmark
    
    compare_hand_averages(left_hand, right_hand)
    
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
release_keys()
