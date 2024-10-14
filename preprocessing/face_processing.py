from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, cv2
#import dlib

def process_and_augmentation_face_data():
    train_datagen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.1,        # Randomly shift the width of images by up to 10%
        height_shift_range = 0.1,       # Randomly shift the height of images by up to 10%
        horizontal_flip = True,         # Flip images horizontally at random
        rescale = 1./255,               # Rescale pixel values to be between 0 and 1
        zoom_range = 0.1,
        shear_range=0.2,
        validation_split = 0.2,
    )

    validation_datagen = ImageDataGenerator(
        rescale = 1./255,               # Rescale pixel values to be between 0 and 1
        validation_split = 0.2          # Set aside 20% of the data for validation
    )

    test_datagen = ImageDataGenerator(
        rescale = 1./255,               # Rescale pixel values to be between 0 and 1
    )

    return train_datagen, validation_datagen, test_datagen

def process_split_face_data(data_path, train_datagen, validation_datagen, test_datagen, input_shape, batch_size):

    train_generator = train_datagen.flow_from_directory(
        directory = f'{data_path}/train',    # Directory containing the training data
        target_size = input_shape[:2],          # Resizes all images to 48x48 pixels
        batch_size = batch_size,                 # Number of images per batch
        color_mode = "grayscale",        # Converts the images to grayscale
        class_mode = "categorical",      # Classifies the images into 7 categories
        subset = "training",              # Uses the training subset of the data
        shuffle = True,
        seed = 12
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory = f'{data_path}/train',     # Directory containing the validation data
        target_size = input_shape[:2],          # Resizes all images to 48x48 pixels
        batch_size = batch_size,                 # Number of images per batch
        color_mode = "grayscale",        # Converts the images to grayscale
        class_mode = "categorical",      # Classifies the images into 7 categories
        subset = "validation",            # Uses the validation subset of the data
        shuffle = True,
        seed = 12
    )

    test_generator = test_datagen.flow_from_directory(
        directory = f'{data_path}/test',     # Directory containing the validation data
        target_size = input_shape[:2],          # Resizes all images to 48x48 pixels
        batch_size = batch_size,                 # Number of images per batch
        color_mode = "grayscale",        # Converts the images to grayscale
        class_mode = "categorical",      # Classifies the images into 7 categories
        subset = "validation",            # Uses the validation subset of the data
        shuffle = False
    )

    return train_generator, validation_generator, test_generator

def clear_screenshots():
    ss = os.listdir("data/screenshots/")
    for image in ss:
        print("Remove: ", "data/screenshots/"+image)
        os.remove("data/screenshots/"+image)

def get_landmarks_from_image(image, detector, predictor):
    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 0)

    landmarks = []
    if len(faces) != 0:
        for i in range(len(faces)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, faces[i]).parts()])
        
    return landmarks

def get_landmarks_from_frame(frame, detector, predictor, show):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 0)

    landmarks = []
    if len(faces) != 0:
        for i in range(len(faces)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, faces[i]).parts()])

            for idx, point in enumerate(landmarks):

                    pos = (point[0, 0], point[0, 1])

                    cv2.circle(frame, pos, 2, color=(255, 255, 255))
                    cv2.putText(frame, str(idx + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, "Faces: " + str(len(faces)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        return None
        
    return landmarks

def preprocess_landmarks(landmarks):
    """
    Preprocess facial landmarks for CNN input
    Args:
        landmarks: numpy matrix of shape (68, 2) containing (x,y) coordinates
        target_size: desired input size for CNN (width, height)
    Returns:
        preprocessed_landmarks: numpy array ready for CNN input
    """
    if len(landmarks) == 0:
        return None
        
    # 1. Convert landmarks to numpy array if not already
    landmarks_array = np.array(landmarks)
    
    # 2. Normalize coordinates to handle different image sizes
    # Get bounding box of face
    min_x, min_y = np.min(landmarks_array, axis=0)
    max_x, max_y = np.max(landmarks_array, axis=0)
    
    # Scale coordinates to range [0,1]
    normalized_landmarks = np.zeros_like(landmarks_array, dtype=np.float32)
    normalized_landmarks[:, 0] = (landmarks_array[:, 0] - min_x) / (max_x - min_x)
    normalized_landmarks[:, 1] = (landmarks_array[:, 1] - min_y) / (max_y - min_y)
    
    # 3. Flatten the landmarks into a 1D array
    flattened_landmarks = normalized_landmarks.flatten()
    
    # 4. Add batch dimension for CNN input
    preprocessed_landmarks = np.expand_dims(flattened_landmarks, axis=0)
    
    return preprocessed_landmarks