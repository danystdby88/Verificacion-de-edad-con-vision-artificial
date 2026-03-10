
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def load_train(path):
    
    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    """
    
    df = pd.read_csv(path)

    # Generador con aumentación para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,     # 80% train, 20% validation
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Flujo de imágenes desde el dataframe
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory='faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',          # regresión → valores continuos
        subset='training',
        shuffle=True
    )


    return train_gen_flow


def load_test(path):

    # Cargar etiquetas
    df = pd.read_csv(path)

    # Generador SOLO para validación (sin aumentación)
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Flujo de imágenes para validación
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=df,
        directory='faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',      # regresión
        subset='validation',
        shuffle=False          # validación → no se mezcla
    )

    return test_gen_flow


def create_model(input_shape):
   

    # Cargar ResNet50 sin la parte superior (top)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Congelar capas base para entrenamiento inicial
    base_model.trainable = False

    # Construcción del modelo final
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)   # salida continua para regresión de edad
    ])

    # Compilación del modelo
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):
    """
    Entrena el modelo dados los parámetros.
    Utiliza generadores de entrenamiento y validación.
    """

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2
    )

    return model



if __name__ == '__main__':

    # Rutas del dataset
    labels_path = 'faces/labels.csv'
    
    # Cargar datos
    train_data = load_train(labels_path)
    test_data = load_test(labels_path)

    # Crear modelo
    model = create_model((224, 224, 3))

    # Entrenar modelo
    model = train_model(model, train_data, test_data, epochs=20)

    # Guardar modelo entrenado
    model.save('age_prediction_model.h5')

    print("Entrenamiento completado y modelo guardado.")
