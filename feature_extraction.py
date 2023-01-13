import tensorflow as tf
import numpy as np
from math import pi

N = 320  # Frame length
M = 160  # Number of overlapping samples
K = 30  # Number of triangular bandpass filters
N_MFCC = 10
alpha = tf.constant(0.96, dtype=tf.float32)
SAMPLE_RATE = 24000
N_FRAMES = 200


def mel_scale(freqs):
    mfreqs = tf.TensorArray(tf.float32, size=tf.shape(freqs)[0], dynamic_size=False)
    i = 0
    for f in freqs:
        mfreqs = mfreqs.write(i, 1127 * tf.math.log(1 + f / 700))
        i += 1
    return mfreqs.stack()


def t_filter_gain(f, s, m, e):
    if f <= s or f >= e:
        return 0.0
    elif f <= m:
        return (f - s) / (m - s)
    else:
        return (e - f) / (e - m)


def get_dct(le):
    return tf.signal.dct(input=le, type=2, n=N_MFCC)


w = tf.TensorArray(tf.float32, size=N, dynamic_size=False)
for n in range(N):
    w = w.write(n, 0.54 - 0.46 * tf.math.cos(2 * pi * n / (N - 1)))
hamming_window = w.stack()

frequencies = tf.constant(np.fft.rfftfreq(N, d=1 / SAMPLE_RATE), dtype=tf.float32)
mel_freqs = mel_scale(frequencies)
bands = tf.linspace(0.0, mel_freqs[-1], K + 2)
filters = tf.TensorArray(tf.float32, size=K, dynamic_size=False)
for fn in range(K):
    filt = tf.TensorArray(tf.float32, size=tf.shape(frequencies)[0], dynamic_size=False)
    i_ = 0
    for f_ in mel_freqs:
        filt = filt.write(i_, t_filter_gain(f_, bands[fn], bands[fn + 1], bands[fn + 2]))
        i_ += 1
    filters = filters.write(fn, filt.stack())

filters = filters.stack()


@tf.function
def frame_blocking(x):
    L = tf.shape(x)[0]
    # y_pre = tf.concat([tf.expand_dims(x[0], axis=0), x[:-1]], axis=0)
    # y_pre_emp = tf.multiply(alpha, y_pre)
    # y = tf.subtract(x, y_pre_emp)
    y = x

    frames = tf.TensorArray(tf.float32, size=N_FRAMES, dynamic_size=False)
    n_frames = tf.constant(0)
    index = tf.constant(0)
    c = lambda fr, nf, i: i + N <= L and nf < N_FRAMES
    b = lambda fr, nf, i: (fr.write(nf, y[i:i + N]),
                           tf.add(nf, 1),
                           tf.add(i, M))
    frames, n_frames, index = tf.while_loop(cond=c, body=b, loop_vars=[frames, n_frames, index])

    c = lambda: (frames.write(n_frames, tf.concat([y[index:L], tf.zeros(N - (L - index))], axis=0)),
                 tf.add(n_frames, 1))
    frames, n_frames = tf.cond(n_frames != N_FRAMES, true_fn=c, false_fn=lambda: (frames, n_frames))

    c = lambda fr, nf: nf < N_FRAMES
    b = lambda fr, nf: (fr.write(nf, tf.zeros(N)), tf.add(nf, 1))
    frames, n_frames = tf.while_loop(cond=c, body=b, loop_vars=[frames, n_frames])

    return frames.stack()


@tf.function
def get_coefficients(frames):
    coefficients = tf.TensorArray(tf.float32, size=N_FRAMES, dynamic_size=False)
    frame_index = 0
    for frame in tf.unstack(frames, axis=0, num=N_FRAMES):
        hw_frame = tf.multiply(frame, hamming_window)
        spectrum = tf.abs(tf.signal.rfft(hw_frame)) ** 2
        E = tf.squeeze(tf.matmul(filters, tf.expand_dims(spectrum, axis=1)), axis=1)
        LE = tf.math.log(tf.add(E, 1e-15))
        coeffs = tf.py_function(get_dct, [LE], tf.float32)
        coefficients = coefficients.write(frame_index, coeffs)
        frame_index += 1

    return coefficients.stack()


@tf.function
def dynamic_coefficients(coeffs):
    pre = tf.concat([tf.zeros((1, tf.shape(coeffs)[1])), coeffs[:-1]], axis=0)
    delta = tf.subtract(coeffs, pre)
    return tf.concat([coeffs, delta], axis=1)


@tf.function
def mfcc(data):
    frames = frame_blocking(data)
    s_coeff = get_coefficients(frames)
    # coefficients = dynamic_coefficients(s_coeff)
    return s_coeff


def calc_mfcc_single_frame(frame):
    frame = tf.cast(frame, dtype=tf.float32)
    hw_frame = tf.multiply(frame, hamming_window)
    spectrum = tf.abs(tf.signal.rfft(hw_frame)) ** 2
    E = tf.squeeze(tf.matmul(filters, tf.expand_dims(spectrum, axis=1)), axis=1)
    LE = tf.math.log(tf.add(E, 1e-15))
    return tf.expand_dims(tf.signal.dct(input=LE, type=2, n=N_MFCC), axis=0)
