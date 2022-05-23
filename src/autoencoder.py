import keras
from keras import layers

encoding_dim = 32
input_img = keras.Input(shape=(784,))

encoded = layers.Dense(encoding_dim, activation='linear')(input_img)
decoded = layers.Dense(784, activation='linear')(encoded)

autoencoder = keras.Model(input_img, decoded)

print("test")