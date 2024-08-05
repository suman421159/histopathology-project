import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_generators(data_dir):
    # Setup data generators with correct parameters
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

    datagen_val = ImageDataGenerator(rescale=1./255)
    datagen_test = ImageDataGenerator(rescale=1./255)

    # Load labels
    labels_csv = os.path.join(data_dir, 'train_labels.csv')
    labels = pd.read_csv(labels_csv)
    labels['label'] = labels['label'].astype(str)
    labels['id'] = labels['id'].apply(lambda x: f"{x}.tif")

    # Split data
    train_labels, val_labels = train_test_split(labels, test_size=0.1, random_state=42, stratify=labels['label'])

    # Create training generator
    train_generator = datagen_train.flow_from_dataframe(
        dataframe=train_labels,
        directory=os.path.join(data_dir, 'train'),
        x_col='id',
        y_col='label',
        target_size=(96, 96),
        batch_size=32,
        class_mode='binary'
    )

    # Create validation generator
    val_generator = datagen_val.flow_from_dataframe(
        dataframe=val_labels,
        directory=os.path.join(data_dir, 'train'),
        x_col='id',
        y_col='label',
        target_size=(96, 96),
        batch_size=32,
        class_mode='binary'
    )

    # Test directory check and generator
    test_path = os.path.join(data_dir, 'test')
    if os.path.exists(test_path) and os.listdir(test_path):
        print("Test directory contents:", os.listdir(test_path))
        test_generator = datagen_test.flow_from_directory(
            directory=test_path,
            target_size=(96, 96),
            batch_size=32,
            class_mode=None,
            shuffle=False
        )
    else:
        print("Test directory is empty or does not exist.")
        test_generator = None

    return train_generator, val_generator, test_generator

# Running the function
data_dir = './data'
train_generator, val_generator, test_generator = setup_generators(data_dir)
if test_generator:
    print("Test batches:", test_generator.n // test_generator.batch_size)
else:
    print("No test batches to display.")
