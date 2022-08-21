import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as pt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images,test_labels) = fashion_mnist.load_data()

pt.imshow(train_images[0],cmap='gray',vmin=0,vmax=255)
pt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5)
test_loss=model.evaluate(test_images,test_labels)

predictions= model.predict(test_images)
print(predictions[0])
print(list(predictions[0]).index(max(predictions[0])))
print(test_labels[0])
