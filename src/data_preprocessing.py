import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def setup_generator(data_dir):
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
    )\
    labels_csv = os.path.join(data_dir, 'train_labels.csv')
    labels = pd.read_csv(labels_csv)
    labels['filename'] = labels['id'] + '.tif'

    train_generator = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=os.path.join(data_dir, 'train'),
        x_col='filename',
        y_col='label',
        target_size=(96, 96),
        class_mode='binary',
        batch_size=32
    )
    return train_generator

if __name__ == '__main__':
    data_dir = './data'
    train_generator = setup_generator(data_dir)

