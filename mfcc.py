import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
import sys
import math
from scipy.fftpack import dct
sys.path.append("..")
from config import cfg


def extract_mfcc(input_signal, sample_rate):
    # 预加重 y(t)=x(t)−αx(t−1)
    emphasized_signal = np.append(input_signal[0], input_signal[1:] - cfg.mfcc.pre_emphasis * input_signal[:-1])
    # 分帧
    frame_size = cfg.mfcc.frame_length_ms * 0.001
    frame_stride = cfg.mfcc.frame_shift_ms * 0.001
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate

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

    # 加窗
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1)) # Explicit Implementation **

    # 傅里叶变换np
    # 傅立叶变换和功率谱
    NFFT = 256
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    # print(mag_frames.shape)
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power Spectrum

    # 三角滤波
    low_freq_mel = 0
    # 将频率转换为Mel
    nfilt = 40  # mel滤波器组：40个滤波器
    # high_freq_mel = (2595 * math.log10(6.714285))
    high_freq_mel = (2595 * math.log10(1 + (sample_rate / 2) / 700))  # 2146
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

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
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
    # 用离散余弦变换（DCT）对滤波器组系数去相关处理，并产生滤波器组的压缩表示
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # 保持在2-13

    # 将正弦升降1应用于MFCC以降低已被声称在噪声信号中改善语音识别的较高MFCC.
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    return np.around(mfcc)


