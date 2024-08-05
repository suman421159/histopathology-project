import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_generator(data_dir):
    labels_csv = os.path.join(data_dir, 'train_labels.csv')
    train_image_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"The labels file was not found at the path: {labels_csv}")
    
    if not os.path.isdir(train_image_dir):
        raise FileNotFoundError(f"The image directory was not found at the path: {train_image_dir}")

    labels = pd.read_csv(labels_csv)
    if 'id' not in labels.columns or 'label' not in labels.columns:
        raise ValueError("CSV file must contain 'id' and 'label' columns")

    labels['filename'] = labels['id'].apply(lambda x: f"{x}.tif")
    labels['label'] = labels['label'].astype(str) 
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    train_generator = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=train_image_dir,
        x_col='filename',
        y_col='label',
        target_size=(96, 96),
        class_mode='binary',
        batch_size=32
    )

    return train_generator

if __name__ == '__main__':
    data_dir = './data'
    try:
        train_generator = setup_generator(data_dir)
        print("Generator setup successful. Ready to train model.")
    except Exception as e:
        print(f"An error occurred: {e}")
