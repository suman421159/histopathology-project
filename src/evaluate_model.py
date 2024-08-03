import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define the directory and load models
test_dir = './data/test/images'  # Make sure this is the correct path
models = [load_model(f'path_to_my_model_fold_{i}.h5') for i in range(1, 6)]

# Setup ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    os.path.dirname(test_dir),  # Navigate up to the folder that includes the 'images' folder
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# Predict using loaded models and average the predictions
predictions = [model.predict(test_generator, verbose=1) for model in models]
average_predictions = np.mean(predictions, axis=0)

# Assuming you have loaded your true labels here into y_true
# y_true = load_your_labels_function()  # You need to implement this part based on your dataset

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, average_predictions)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
