import attractiveness
import util
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
import tkinter as tk
from tkinter import filedialog as fd



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


window = tk.Tk()
