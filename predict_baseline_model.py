import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append("..")
from config import cfg
import numpy as np
import scipy.io.wavfile
from mfcc import *
from data_prepare import resample_signal_16_to_8

model = keras.models.load_model(cfg.save_mode_path)

wav_filename = "./test_some/my_wav/yes4.wav"


sample_rate, wave_data = scipy.io.wavfile.read(wav_filename)
resampled_rate, resampled_signal = resample_signal_16_to_8(wave_data, sample_rate, rate=2)
resampled_signal = np.append(resampled_signal, np.zeros(8000 - len(resampled_signal), dtype=np.int16))

mfcc = extract_mfcc(resampled_signal, resampled_rate).astype(np.int64)     # MFCC特征提取

mfcc_features = np.reshape(mfcc, [50, 12])

mfcc_features = (mfcc_features - np.min(mfcc_features)) / (np.max(mfcc_features) - np.min(mfcc_features))

input_wav = tf.expand_dims(mfcc_features, axis=0)
input_wav = tf.expand_dims(input_wav, axis=3)
predictions = model.predict(input_wav)
score = tf.nn.softmax(predictions[0])

predic_cls = np.argmax(score)
predic_scr = 100 * np.max(score)
print(predic_cls)
print(predic_scr)