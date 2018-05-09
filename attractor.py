import attractiveness
import util
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
import tkinter as tk
from tkinter import filedialog as fd
import cv2



model = Sequential()

# This trainable layer represents an image that we will be modifying
# It always receives exactly 1 as its input, and so produces its weights as its output
# The weights are just the flattened image
image_layer = Dense(util.shape_volume(attractiveness.image_shape), use_bias = False, input_shape = (1,))
model.add(image_layer)
model.add(Reshape(attractiveness.image_shape))

# Now take the image and estimate its attractiveness without changing the estimator
attractiveness.model.trainable = False
model.add(attractiveness.model)
model.compile(optimizer = 'adam', loss = 'mse')


def get_image(dims = None):
    weights = image_layer.get_weights()
    reshaped = np.reshape(weights, attractiveness.image_shape)
    return util.cut_resize_image(reshaped, dims)


def set_image(image):
    cut_resized = util.cut_resize_image(image, attractiveness.image_dims)
    reshaped = np.reshape(cut_resized, (1, 1, util.shape_volume(attractiveness.image_shape)))
    image_layer.set_weights(reshaped)


def get_attractiveness():
    batch_prediction = model.predict_on_batch([1.0])
    return batch_prediction[0][0]


def train_attractiveness(target):
    model.train_on_batch([1.0], [target])


set_image(util.test_faces[0].get_image(attractiveness.image_dims))


def manual_interface():
    preview_name = 'Image'
    slider_length = 256
    target_attractiveness = get_attractiveness()

    control_window = tk.Tk()
    control_window.title('Control')
    current_slider = tk.Scale(control_window, from_ = 0, to = 1, resolution = 0.01,
    orient = tk.HORIZONTAL, length = slider_length, label = 'Current attractiveness')
    current_slider.pack()
    target_slider = tk.Scale(control_window, from_ = 0, to = 1, resolution = 0.01,
    orient = tk.HORIZONTAL, length = slider_length, label = 'Target attractiveness')
    target_slider.set(target_attractiveness)
    target_slider.pack()

    closed = False
    while not closed:
        try:
            current_slider.set(get_attractiveness())
            target_attractiveness = target_slider.get()
            train_attractiveness(target_attractiveness)
            control_window.update()
        except:
            closed = True
        preview(window_name = preview_name, blocking = False)
        if cv2.getWindowProperty(preview_name, 0) < 0:
            closed = True

    try:
        control_window.destroy()
    except:
        pass
    try:
        cv2.destroyWindow(preview_name)
    except:
        pass


def preview(window_name = 'Image', dims = (512, 512), blocking = True):
    util.preview([window_name], [get_image()], dims = dims,
    destroy = blocking, milliseconds = 0 if blocking else 1)
