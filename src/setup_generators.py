import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_generators(data_dir):
    # Data augmentation configuration for training data
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Data generator for validation data (only rescaling)
    datagen_val = ImageDataGenerator(rescale=1./255)
    
    # Data generator for test data (only rescaling, no labels)
    datagen_test = ImageDataGenerator(rescale=1./255)

    # Load labels and prepare training and validation splits
    labels_csv = os.path.join(data_dir, 'train_labels.csv')
    labels = pd.read_csv(labels_csv)
    labels['label'] = labels['label'].astype(str)
    labels['id'] = labels['id'].apply(lambda x: f"{x}.tif")

    train_labels, val_labels = train_test_split(labels, test_size=0.1, random_state=42, stratify=labels['label'])

    # Setup training data generator
    train_generator = datagen_train.flow_from_dataframe(
        dataframe=train_labels,
        directory=os.path.join(data_dir, 'train'),
        x_col='id',
        y_col='label',
        target_size=(96, 96),
        batch_size=32,
        class_mode='binary'
    )

    # Setup validation data generator
    val_generator = datagen_val.flow_from_dataframe(
        dataframe=val_labels,
        directory=os.path.join(data_dir, 'train'),
        x_col='id',
        y_col='label',
        target_size=(96, 96),
        batch_size=32,
        class_mode='binary'
    )

    # Setup test data generator
    test_generator = datagen_test.flow_from_directory(
        directory=os.path.join(data_dir, 'test/images'),
        target_size=(96, 96),
        batch_size=32,
        class_mode=None,  # Important for test data as there are no labels
        shuffle=False      # Keep data in order to match the predictions with IDs or filenames
    )

    return train_generator, val_generator, test_generator

# Assuming setup_generators function is being called here
data_dir = './data'  # or wherever your data directory is
train_generator, val_generator, test_generator = setup_generators(data_dir)

# Printing out the number of batches to verify correct setup
print("Training batches:", train_generator.n // train_generator.batch_size)
print("Validation batches:", val_generator.n // val_generator.batch_size)
print("Test batches:", test_generator.n // test_generator.batch_size)
