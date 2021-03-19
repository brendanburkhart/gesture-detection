import random
from hashlib import sha256

import numpy as np
import cv2
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

"""
Keras model input image shape
"""
model_input_shape = (80, 120, 1)

"""
Image load options for Keras image data generator
"""
load_options = {
    "class_mode": 'categorical',
    "color_mode": "grayscale",
    "target_size": (model_input_shape[0], model_input_shape[1]),
    "batch_size": 32,
}

"""
With 50% probability, randomly inverts image.
"""
def random_image_invert(image):
    invert = random.randint(0, 1) == 1
    return 255 - image if invert else image

"""
Randomly shuffle channels
"""
def random_channel_shuffle(image):
    channel_order = [0, 1, 2]
    random.shuffle(channel_order)

    red = image[:, :, channel_order[0]]
    green = image[:, :, channel_order[1]]
    blue = image[:, :, channel_order[2]]

    rgb = (red[..., np.newaxis], green[..., np.newaxis], blue[..., np.newaxis])
    return np.concatenate(rgb, axis=-1)

"""
Perform color augmentations
"""
def augment_color(image):
    shuffled = random_channel_shuffle(image)
    inverted = random_image_invert(shuffled)

    return inverted

"""
Creates Keras ImageDataGenerator to augment data
"""
def prepare_data():
    return ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=90,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.7, 1.3],
        shear_range=0.3,
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.2,
    )

"""
Removes modifiers from OpenCV key code, such as control,
shift, alt modifiers.
"""
def key(key_code):
    return key_code % 2**16

"""
Does OpenCV key code represent enter key
"""
def is_enter(key_code):
    return key(key_code) == 13

"""
Does OpenCV key code represent escape key
"""
def is_escape(key_code):
    return key(key_code) == 27

"""
Resize and crop raw camera capture to be 300x200 without distorting it.

First resize image so width is at least 300 and height is at least 200 while
maintaining the original aspect ratio. Then crop it down to 300x200.
"""
def resize_raw(image):
    width = int(max(300.0, 200.0*(image.shape[1]/image.shape[0])))
    height = int(300.0*image.shape[0]/image.shape[1])

    resized = cv2.resize(image, (width, height))
    cropped = resized[0:200, 0:300]

    return cropped

"""
25-character hex string of SHA-256 hash of image data
"""
def image_hash(image):
    data = str(image).encode('utf-8')
    return sha256(data).hexdigest()[0:25]

"""
Convert Keras image format to OpenCV image format

Keras loads images in RGB channel order, while OpenCV expects BGR
channel order when displaying images.
"""
def keras_img_to_opencv_mat(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

"""
Convert OpenCV image format to Keras image format

OpenCV loads images in BGR channel order, while Keras expects RGB.
"""
def opencv_mat_to_keras_img(mat):
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    mat = mat.astype("float32") / 255.0
    mat = cv2.resize(mat, (model_input_shape[1], model_input_shape[0]))
    return np.expand_dims(mat, axis=-1)

"""
Loads trained Keras model from specified folder
"""
def load_model(model_folder: str):
    return models.load_model(model_folder)

"""
Gets model's prediction for Keras image
"""
def predict(model, image):
    return model(np.array([image]))[0]

"""
Given prediction, returns predicted label and probability of the more probable
category.
"""
def label(prediction):
    class_names = ['None', 'Paper', 'Rock', 'Scissors']
    index = np.argmax(prediction)
    return (class_names[index], prediction[index])

"""
Returns predicted label and probability given model and OpenCV mat
"""
def classify(model, mat):
    img = opencv_mat_to_keras_img(mat)
    prediction = predict(model, img)
    labelled = label(prediction)
    return labelled

"""
Adds label and probability text over Opencv mat
"""
def label_image(name, probability, mat):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    text = "{}: {}%".format(name, int(probability*100))
    x,y = 25, mat.shape[0] - 25
    return cv2.putText(mat, text, (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

"""
Creates new OpenCV window with specified name and size
"""
def create_window(name, size):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size)
