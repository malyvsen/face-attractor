import util
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Conv2D, Dense
from keras.layers import MaxPooling2D, Flatten
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN



image_dims = (128, 128)
num_channels = 3
image_shape = (image_dims[0], image_dims[1], num_channels)
batch_size = 32
model_path = 'models/attractiveness.h5'
log_dir = './logs/attractivenesss'


model = Sequential()

model.add(Conv2D(32, kernel_size = 5, padding = 'same', input_shape = image_shape))
model.add(MaxPooling2D())
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size = 5, padding = 'same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size = 5, padding = 'same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')


try:
    model.load_weights(model_path)
    print('Attractiveness model loaded from ' + model_path)
except ValueError:
    print('Attractiveness model failed to load - perhaps due to changes in architecture? Starting fresh...')
except OSError:
    print('Attractiveness model failed to load - perhaps file is missing? Starting fresh...')


train_generator = util.generator(util.attractiveness_batch, batch_size = batch_size, dims = image_dims, test = False)
test_generator = util.generator(util.attractiveness_batch, dims = image_dims, test = False)


def train(epochs = 64):
    model.fit_generator(train_generator,
    epochs = epochs, steps_per_epoch = util.set_steps(batch_size = batch_size, test = False),
    validation_data = test_generator, validation_steps = len(util.test_faces),
    callbacks = [ModelCheckpoint(filepath = model_path, save_best_only = True, save_weights_only = True),
    TensorBoard(log_dir = log_dir),
    TerminateOnNaN()])


def test():
    return model.evaluate_generator(test_generator, steps = 1)
