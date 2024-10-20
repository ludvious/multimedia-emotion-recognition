import os
import numpy as np
from preprocessing.face_processing import get_landmarks_from_image, preprocess_landmarks

def load_landmarks_and_labels(data_path, landmark_detector, landmark_predictor, output_folder='data/face_landmarks/'):
    '''
    method for create features and labels with landmarks from face data path
    '''
    features = []
    labels = []
    emotion_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    for _, folder in enumerate(emotion_folders):
        folder_dir = os.path.join(data_path, folder)
        if os.path.isdir(folder_dir):
            for emotion in os.listdir(folder_dir):
                emotion_dir = os.path.join(folder_dir, emotion)
                for file_path in os.listdir(emotion_dir):
                    image_path = os.path.join(emotion_dir, file_path)
                    face_landmarks = get_landmarks_from_image(image_path, landmark_detector, landmark_predictor)
                    processed_landmarks = preprocess_landmarks(face_landmarks)
                    if processed_landmarks is not None:
                        features.append(processed_landmarks)
                        labels.append(emotion)
                print(f"Processed {len(features)} features for emotion: {emotion}")
    # Save the processed data
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    np.save(os.path.join(output_folder, 'X_landmarks.npy'), X)
    np.save(os.path.join(output_folder, 'y_labels.npy'), y)

    print(f"Processed data saved to {output_folder}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y


# Load train and test features
'''train_dir = "/content/train"
val_dir = "/content/test"
train_features, train_labels = load_features_and_labels(train_dir)
val_features, val_labels = load_features_and_labels(val_dir)'''

#train_landmark = get_face_landmarks(train_features)
#val_landmark = get_face_landmarks(val_features)