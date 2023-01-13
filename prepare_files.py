import os
import wave
import numpy as np


audio = wave.open('voice_24khz/spec.wav', mode='rb')
byte_frames = audio.readframes(-1)
data = np.frombuffer(byte_frames, dtype=np.int16)
print(len(data))
if len(data) < 32000:
    data = np.concatenate([data, np.zeros(32000 - len(data), dtype=np.int16)])
# data = np.zeros(32000, dtype=np.int16)

for i in range(7000):
    ndata = data[:32000].copy()
    inds = np.random.randint(0, 32000, 2, dtype=int)
    min_i, max_i = min(inds), max(inds)
    noise_level = np.random.randint(200, 2500, 1, dtype=int)
    noise = np.floor(np.random.randn(max_i - min_i) * noise_level).astype(np.int16)
    ndata[min_i:max_i] += noise

    out = wave.Wave_write(f'data/SPEC/spec_{i}.wav')
    out.setframerate(24000)
    out.setnchannels(1)
    out.setsampwidth(2)
    out.writeframes(ndata.tobytes())
    out.close()
