# Rock-paper-scissors gesture detection

## Installing

You will need to first install Python, and then install the Python packages OpenCC, Numpy, and Keras. OpenCV is a computer vision library, which is used here for capture and processing of images. Numpy is a popular numerical processing library, and Keras is a higher-level machine learning library built on top of Tensorflow. Tensorflow currently only supports Python versions 3.5–3.8, so if you have Python 3.9 installed you will need to create a separate installation or downgrade to v3.8.

## CNN

This project uses a convolutional neural network, a type of neural net specifically designed to work well with images. The network is constructed in the `train.py` file. More can be read about CNNs and Keras layers elsewhere, but for a brief overview, a network is built out of a series of layers.

- The most important layers in a Keras CNN are Conv2D layers, these perform the actual convolutions.
- MaxPooling2D layers down-sample the amoumt of data flowing through that layer of the let -- since we want the network to be extracting higher and higher-level descriptors in each layer, we force it to keep fewer data.
- Dropout layers just pass data through from the previous layer, but will randomly fail to pass through some data, this helps prevent overfitting.
- A Flatten layer converts the data from a 3D image tensor to a 1D vector.
- Dense layers are standard fully-connected neural network layers, used to make the final classification.

Almost all of the training data features hands in evenly lit environments on a green background, and are all captured from the same angle. In order to create a trained model that works with a wider variety of lighting conditions, skin tones, background colors, and camera angles, the data is _augmented_. This means every time a training image is used, it is randomly stretched, rotated, flipped, re-colored, brightened/darkened, etc. This also allows each image to be re-used many times during training, since the data augmentation process will produce an entirely different processed image every time.

## Usage

There are three different main scripts in this project. All of them use the `utils.py` file which contains a variety of common functionality.

- `preview.py` grabs and displays random training images after they gone through the augmentation process.
- `train.py` will re-train the CNN on the training data in the specified directory. For example, run `python train.py ./data` to re-train on images in the `/data` folder.
- `rps.py` will start an OpenCV camera capture and attempt to classify frames in the video feed as rock, paper, or scissors.
