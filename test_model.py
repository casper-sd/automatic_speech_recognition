import keras
import numpy as np
import sounddevice as sd
from feature_extraction import calc_mfcc_single_frame
from threading import Thread
from queue import Queue
import tensorflow as tf
import soundfile as sf
import time

N_PREDICTORS = 5
OFFSET = 40
commands = ['bluetooth', 'default', 'invalid', 'radio', 'spec']
model = keras.models.load_model('sr_model.h5')
model.summary()

duration = 1000
test_frames = Queue(maxsize=250)
p_queues = [Queue(maxsize=250) for _ in range(N_PREDICTORS)]
prev_fr = np.zeros(160)


def callback(indata, outdata, frames, time, status):
    global prev_fr
    f = np.squeeze(indata, axis=1)
    fr = np.concatenate((prev_fr, f), axis=0)
    test_frames.put(fr)
    prev_fr = f
    outdata[:] = indata


def get_live_audio():
    with sd.Stream(channels=1, samplerate=16000, dtype=np.float32, callback=callback, blocksize=160):
        sd.sleep(int(duration * 1000))


def predict(ind):
    while True:
        features = []
        for _ in range(200):
            features.append(p_queues[ind].get())

        features = tf.concat(features, axis=0)
        features = features[tf.newaxis, ..., tf.newaxis]
        prediction = model.predict(features)[0]
        ii = np.argmax(prediction)
        print(commands[ii] + ':' + str(prediction[ii]))


live = Thread(target=get_live_audio)
live.start()

for i in range(N_PREDICTORS):
    Thread(target=predict, args=(i,)).start()
    for _ in range((N_PREDICTORS - i - 1)*OFFSET):
        p_queues[i].put(tf.zeros((1, 10)))

while live.is_alive():
    frame = test_frames.get()
    mfcc_frame = calc_mfcc_single_frame(frame)
    for i in range(N_PREDICTORS):
        p_queues[i].put(mfcc_frame)


print('done')
