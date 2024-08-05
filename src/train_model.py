from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def create_model(input_shape):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    data_dir = './data'
    labels_csv = f'{data_dir}/train_labels.csv'
    labels = pd.read_csv(labels_csv)
    labels['id'] = labels['id'].apply(lambda x: f"{x}.tif")
    labels['label'] = labels['label'].astype(str)
    
    datagen = ImageDataGenerator(rescale=1./255)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_var = 1

    for train_index, val_index in skf.split(np.zeros(len(labels)), labels['label']):
        training_data = labels.iloc[train_index]
        validation_data = labels.iloc[val_index]

        train_generator = datagen.flow_from_dataframe(
            dataframe=training_data,
            directory=f'{data_dir}/train',
            x_col='id',
            y_col='label',
            target_size=(96, 96),
            batch_size=32,
            class_mode='binary'
        )

        validation_generator = datagen.flow_from_dataframe(
            dataframe=validation_data,
            directory=f'{data_dir}/train',
            x_col='id',
            y_col='label',
            target_size=(96, 96),
            batch_size=32,
            class_mode='binary'
        )

        model = create_model((96, 96, 3))
        
        # Setup checkpointing
        checkpoint_callback = ModelCheckpoint(
            f'path_to_my_model_fold_{fold_var}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        print(f"Training for fold {fold_var} ...")
        history = model.fit(
            train_generator,
            steps_per_epoch=100,
            validation_data=validation_generator,
            validation_steps=50,
            epochs=10,
            callbacks=[checkpoint_callback]  
        )
        
        fold_var += 1
