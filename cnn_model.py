import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import cfg
from tensorflow.keras import regularizers

def sequential_model():
    # initialize the model along with the input shape to be
    # "channels last" ordering
    input_wav = keras.Input(shape=(50, 12, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_wav)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    model_output = layers.Dense(cfg.num_classes)(x)
    model = keras.Model(input_wav, model_output)
    # return the constructed network architecture
    return model


# mymodel = sequential_model()
# mymodel.summary()
