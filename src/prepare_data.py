import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (96, 96))  
        img = img / 255.0  
        return img
    else:
        return None

def setup_data_generator():
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True
    )
    return datagen

def main():
    data_dir = './data'
    labels_csv = os.path.join(data_dir, 'train_labels.csv')
    
    try:
        labels = pd.read_csv(labels_csv)
        print(f"Loaded {len(labels)} labels.")
    except FileNotFoundError:
        print(f"Failed to load labels from {labels_csv}. Please check the file path.")
        return
    
    train_dir = os.path.join(data_dir, 'train')  # Path where the images are stored
    try:
        # List all files in the train directory and filter out non-image files
        image_files = [f for f in os.listdir(train_dir) if f.endswith('.tif')]
        if image_files:
            sample_image_path = os.path.join(train_dir, image_files[0])
            sample_image = load_image(sample_image_path)
            if sample_image is not None:
                plt.imshow(sample_image)
                plt.title('Sample Image')
                plt.show()
            else:
                print(f"Failed to load image from {sample_image_path}")
        else:
            print("No image files found in the directory.")
    except Exception as e:
        print(f"Error during image loading or display: {e}")

if __name__ == '__main__':
    main()
