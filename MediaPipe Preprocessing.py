import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define dataset path
dataset_path = "dataset"
classes = sorted(os.listdir(dataset_path))  # Assumes folders are named by class

data = []  # Stores extracted landmark features
labels = []  # Stores corresponding labels

# Process each class folder
for class_idx, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_path, class_name)
    image_files = [f for f in os.listdir(class_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
        image_path = os.path.join(class_folder, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark  # Only one hand per image
            
            # Extract X, Y coordinates
            landmark_features = []
            for lm in landmarks:
                landmark_features.extend([lm.x, lm.y])
            
            data.append(landmark_features)
            labels.append(class_idx)

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("hand_gesture_data.csv", index=False)

print("Data preprocessing complete. Saved as hand_gesture_data.csv")
