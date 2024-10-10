from preprocessing.face_processing import process_and_augmentation_face_data, process_split_face_data, get_landmarks_from_image, preprocess_landmarks
from preprocessing.audio_processing import AudioProcessing
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
import numpy as np
from pathlib import Path
import cv2, os
from config import N_MELS_BAND

def create_face_dataset(data_path: str, input_shape, batch_size):

    train_datagen, validation_datagen, test_datagen = process_and_augmentation_face_data()
    train_gen, val_gen, test_gen = process_split_face_data(data_path, train_datagen, validation_datagen, input_shape, batch_size)

    return train_gen, val_gen

def load_features_and_labels(path):
    '''
    method for create features and labels with landmarks from face data path
    '''
    features = []
    labels = []
    emotions = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    
    for emotion in emotions.keys():
        emotion_dir = os.path.join(path, emotion)
        if os.path.isdir(emotion_dir):
            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (48, 48))  # Resize image to 48x48 pixels
                face_landmarks = get_landmarks_from_image(image)
                processed_landmarks = preprocess_landmarks(face_landmarks)
                if face_landmarks is not None:
                    features.append(processed_landmarks)
                    labels.append(emotions[emotion])
            print(f"Processed {len(features)} features for emotion: {emotion}")
    
    return np.array(features), np.array(labels)
    
def create_audio_dataset(audio_file_path: str):
        """
        Load all audio wav files from folders, extract features, and return the dataset with features and labels.

        Returns:
            np.array: Features and corresponding labels.
        """
        audio_proc = AudioProcessing()
        X = [] # features
        Y = [] # labels

        audio_path = Path(audio_file_path)
        for label in audio_path.iterdir(): #each folder name must be the label name
            if label.is_dir():
                for audio_file in label.iterdir():
                    file_path = audio_file
                    try:
                        # Load and preprocess audio
                        audio = audio_proc.load_audio(file_path)
                        # Extract Mel spectrogram features
                        mel_spectrogram = audio_proc.get_spectrogram(audio)
                        feature = np.expand_dims(mel_spectrogram, axis=-1)
                        # Append the features and label
                        X.append(feature)
                        Y.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return np.array(X), np.array(Y)

def process_audio_data_generator(data_path, batch_size, input_shape):

        train_gen = ImageDataGenerator(
            rescale = 1./255,                                  # Rescale pixel values to be between 0 and 1
            validation_split = 0.2                             # Set aside 20% of the data for validation
        )

        val_gen = ImageDataGenerator(
            rescale = 1./255,                                  # Rescale pixel values to be between 0 and 1
            validation_split = 0.2                             # Set aside 20% of the data for validation
        )
        
        train_generator = train_gen.flow_from_directory(
            directory = data_path,                             # Directory containing the training data
            target_size = input_shape[:2],                     # Resizes all images to 48x48 pixels
            batch_size = batch_size,                           # Number of images per batch
            color_mode = "rgb",                                # Converts the images to grayscale
            class_mode = "categorical",                        # Classifies the images into 7 categories
            subset = "training",                               # Uses the training subset of the data
            shuffle = True,
            seed = 0
        )

        val_generator = val_gen.flow_from_directory(
            directory = data_path,                             # Directory containing the training data
            target_size = input_shape[:2],                     # Resizes all images to 48x48 pixels
            batch_size = batch_size,                           # Number of images per batch
            color_mode = "rgb",                                # Converts the images to grayscale
            class_mode = "categorical",                        # Classifies the images into 7 categories
            subset = "validation",                             # Uses the training subset of the data
            shuffle = True,
            seed = 0
        )

        return train_generator, val_generator


# Load train and test features
train_dir = "/content/train"
val_dir = "/content/test"
train_features, train_labels = load_features_and_labels(train_dir)
val_features, val_labels = load_features_and_labels(val_dir)

#train_landmark = get_face_landmarks(train_features)
#val_landmark = get_face_landmarks(val_features)