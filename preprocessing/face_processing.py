from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

def augmentation_face_data():
    train_datagen = ImageDataGenerator(
        width_shift_range = 0.1,        # Randomly shift the width of images by up to 10%
        height_shift_range = 0.1,       # Randomly shift the height of images by up to 10%
        horizontal_flip = True,         # Flip images horizontally at random
        rescale = 1./255,               # Rescale pixel values to be between 0 and 1
        validation_split = 0.2          # Set aside 20% of the data for validation
    )

    validation_datagen = ImageDataGenerator(
        rescale = 1./255,               # Rescale pixel values to be between 0 and 1
        validation_split = 0.2          # Set aside 20% of the data for validation
    )

    return train_datagen, validation_datagen

def process_split_face_data(data_path, train_datagen, validation_datagen, input_shape, batch_size):

    train_generator = train_datagen.flow_from_directory(
        directory = f'{data_path}/train',    # Directory containing the training data
        target_size = input_shape[:2],          # Resizes all images to 48x48 pixels
        batch_size = batch_size,                 # Number of images per batch
        color_mode = "grayscale",        # Converts the images to grayscale
        class_mode = "categorical",      # Classifies the images into 7 categories
        subset = "training"              # Uses the training subset of the data
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory = f'{data_path}/test',     # Directory containing the validation data
        target_size = input_shape[:2],          # Resizes all images to 48x48 pixels
        batch_size = 64,                 # Number of images per batch
        color_mode = "grayscale",        # Converts the images to grayscale
        class_mode = "categorical",      # Classifies the images into 7 categories
        subset = "validation"            # Uses the validation subset of the data
    )

    return train_generator, validation_generator
