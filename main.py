import os
import pathlib
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import mfcc


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_mfcc_and_label_id(waveform, label):
    waveform = tf.cast(waveform, dtype=tf.float32)
    features = mfcc(waveform)
    label_id = tf.argmax(label == commands)
    features = features[..., tf.newaxis]
    return features, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    waveform_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = waveform_ds.map(map_func=get_mfcc_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds


tf.random.set_seed(42)
np.random.seed(42)

labels = []
DATASET_PATH = 'data'
data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
n_samples = len(filenames)
print('Number of total examples:', n_samples)

train_size = int(0.8 * n_samples)
val_size = int(0.1 * n_samples)
test_size = n_samples - train_size - val_size

train_files = filenames[:train_size]
val_files = filenames[train_size:train_size+val_size]
test_files = filenames[-test_size:]

train_ds = mfcc_ds = preprocess_dataset(train_files)
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

# Visualize MFCC
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, (mfccs, l_id) in enumerate(mfcc_ds.take(9)):
    r = i // 3
    c = i % 3
    ax = axes[r][c]
    mfccs = np.squeeze(mfccs, axis=-1)
    height = mfccs.shape[0]
    width = mfccs.shape[1]
    X = np.linspace(0, np.size(mfccs), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, mfccs)
    ax.set_title(commands[l_id.numpy()])
    ax.axis('off')

plt.show()

##################################################################
batch_size = 70
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = ()
for example in mfcc_ds.take(1):
    input_shape = example[0].shape
print('Input shape:', input_shape)
num_labels = len(commands)

# Build Model

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(100, 10),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_labels, activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
)

EPOCHS = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=4),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

test_audio = []
test_labels = []

for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=commands,
            yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

model.save('sr_model.h5')
