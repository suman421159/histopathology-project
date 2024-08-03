import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    labels_csv = os.path.join(data_dir, 'train_labels.csv')
    labels = pd.read_csv(labels_csv)
    return train_dir, labels

def visualize_data(train_dir, labels):
    sns.countplot(x='label', data=labels)
    plt.title('Distribution of Labels')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, label in enumerate(labels['label'].unique()):
        sample_file = labels[labels['label'] == label]['id'].iloc[0] + '.tif'
        img_path = os.path.join(train_dir, sample_file)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f'Sample Image: Label {label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_dir = './data'
    train_dir, labels = load_data(data_dir)
    visualize_data(train_dir, labels)
