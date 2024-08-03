from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

# Define the directory where models are saved
model_dir = './models'  # Adjust this if your models are in a different directory

# Load models
models = [load_model(os.path.join(model_dir, f'path_to_my_model_fold_{i}.h5')) for i in range(1, 6)]

# Setup the ImageDataGenerator for the test set
datagen = ImageDataGenerator(rescale=1./255)

# Assuming your test images are in a subdirectory called 'images' inside the 'test' folder
test_dir = './data/test/images'  
test_generator = datagen.flow_from_directory(
    os.path.dirname(test_dir),  # This points to './data/test', the parent directory of 'images'
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,  # No labels are provided
    shuffle=False  # Important for maintaining order
)

# Verify the setup
if test_generator.n > 0:
    print(f"Found {test_generator.n} images for testing.")
    for model in models:
        # Predict on the entire test data
        predictions = model.predict(test_generator, verbose=1)
        print("Predictions made successfully.")
else:
    print("No images found for testing. Check the directory path and contents.")
