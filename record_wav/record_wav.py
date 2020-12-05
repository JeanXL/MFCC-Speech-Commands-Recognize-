import pyaudio
import wave
from tqdm import tqdm

CHUNK = 160  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1  # 单声道
RATE = 16000  # 采样频率16K

def record_audio(wave_out_path, record_second):
    """ 录音功能 """
    p = pyaudio.PyAudio()  # 实例化对象
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)  # 打开流，传入响应参数
    wf = wave.open(wave_out_path, 'wb')  # 打开 wav 文件。
    wf.setnchannels(CHANNELS)  # 声道设置
    wf.setsampwidth(p.get_sample_size(FORMAT))  # 采样位数设置
    wf.setframerate(RATE)  # 采样频率设置

    print("开始录音：++++++++++")
    for _ in tqdm(range(0, int(RATE * record_second / CHUNK))):
        data = stream.read(CHUNK)
        wf.writeframes(data)  # 写入数据

    print("录音结束：++++++++++")
    stream.stop_stream()  # 关闭流
    stream.close()
    p.terminate()
    wf.close()

record_audio("./record_test.wav", 1)    #录制1s音频，保存record_test.wav
