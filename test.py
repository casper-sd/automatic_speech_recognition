
from keras import layers, models
num_labels = 5
input_shape = (200, 10, 1)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(100, 10),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_labels, activation='softmax')
])

model.summary()