import pathlib
import sys

import utils

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

def train(data_gen, data_dir):
    train_iter = data_gen.flow_from_directory(
        data_dir,
        **utils.load_options,
        subset="training"
    )

    validation_iter = data_gen.flow_from_directory(
        data_dir,
        **utils.load_options,
        subset="validation"
    )

    model = Sequential([
        Conv2D(256, 5, activation='relu', padding='same', input_shape=utils.model_input_shape),
        MaxPooling2D(pool_size=2),
        Conv2D(128, 5, activation='relu', padding='same'),
        MaxPooling2D(pool_size=2),
        Dropout(0.2),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        train_iter,
        epochs=10,
        validation_data=validation_iter,
    )

    return model

def main(data):
    data_gen = utils.prepare_data()

    data_folder = pathlib.Path(data)
    model_folder = pathlib.Path(data).parent / (data_folder.name + "_model")

    model = train(data_gen, str(data_folder))
    model.save(str(model_folder))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please supply data directory as argument")
        exit(-1)

    main(sys.argv[1])
