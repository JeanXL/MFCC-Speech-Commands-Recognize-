import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import math
from scipy.fftpack import dct

def resample_signal_16_to_8(origin_signal, origin_rate, rate=2):
    if rate > len(origin_signal):
        print("# num out of length, return arange:", end=" ")
        return origin_signal
    resampled_signal = np.array([], dtype=origin_signal.dtype)
    seg = int(len(origin_signal) / rate)
    for n in range(seg):
        resampled_signal = np.append(resampled_signal, origin_signal[int(rate * n)])
    return int(origin_rate/rate), resampled_signal

wav_filename = "testno.wav"

# 1. 读取原始音频信号
sample_rate, wave_data = scipy.io.wavfile.read(wav_filename)
resampled_rate, resampled_signal = resample_signal_16_to_8(wave_data, sample_rate, rate=2)
if len(resampled_signal) < 8000:
    input_signal = np.append(resampled_signal, np.zeros(8000 - len(resampled_signal), dtype=np.int16))
elif len(resampled_signal) > 8000:
    input_signal = resampled_signal[0:8000]
else:
    input_signal = resampled_signal

# 2. 预加重 y(t)=x(t)−αx(t−1)  0.97
emphasized_signal = np.append(resampled_signal[0], resampled_signal[1:] - 0.97 * resampled_signal[:-1])

# 3. 分帧
frame_size = 32 * 0.001
frame_stride = 20 * 0.001
frame_length, frame_step = frame_size * resampled_rate, frame_stride * resampled_rate

signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step + 1))

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z)

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

# 4. 加窗
frames *= np.hamming(frame_length)
# frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1)) # Explicit Implementation **

# 5. 傅里叶变换，功率谱
NFFT = 256
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
# print(mag_frames.shape)
pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

# 6. Mel滤波
low_freq_mel = 0
# 将频率转换为Mel
nfilt = 40  # mel滤波器组：40个滤波器
# high_freq_mel = (2595 * math.log10(6.714285))
high_freq_mel = (2595 * math.log10(1 + (resampled_rate / 2) / 700))  # 2146
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / resampled_rate)
fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])  # left
    f_m = int(bin[m])  # center
    f_m_plus = int(bin[m + 1])  # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
mel_energy = 20 * np.log10(filter_banks)  # dB

# 7. 用离散余弦变换（DCT）对滤波器组系数去相关处理，并产生滤波器组的压缩表示
num_ceps = 12
mfcc = dct(mel_energy, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # 保持在2-13

# 8. 将正弦升降应用于MFCC以降低已被声称在噪声信号中改善语音识别的较高MFCC.
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter = 22
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift

# 9. 归一化
mfcc_features = np.reshape(mfcc, [50, 12])
mfcc_features = (mfcc_features - np.min(mfcc_features)) / (np.max(mfcc_features) - np.min(mfcc_features))
print("<<<<<<<<<")