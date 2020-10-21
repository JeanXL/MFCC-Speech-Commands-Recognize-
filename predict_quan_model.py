import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append("..")
import scipy.io.wavfile
from mfcc import *
from data_prepare import resample_signal_16_to_8

wav_filename = "./test_some/my_wav/yes4.wav"

sample_rate, wave_data = scipy.io.wavfile.read(wav_filename)
resampled_rate, resampled_signal = resample_signal_16_to_8(wave_data, sample_rate, rate=2)
resampled_signal = np.append(resampled_signal, np.zeros(8000 - len(resampled_signal), dtype=np.int16))

mfcc = extract_mfcc(resampled_signal, resampled_rate).astype(np.int64)     # MFCC特征提取

mfcc_features = np.reshape(mfcc, [50, 12])

mfcc_features = (mfcc_features - np.min(mfcc_features)) / (np.max(mfcc_features) - np.min(mfcc_features))

input_wav = tf.expand_dims(mfcc_features, axis=2)

def predict_quan_model(tflite_file, test_wav):

    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        test_image = test_wav / input_scale + input_zero_point

    test_image = np.expand_dims(test_wav, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    # Check if the output type is quantized, then rescale output data to float
    if output_details['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details["quantization"]
        test_image = test_image.astype(np.float32)
        test_image = test_image / input_scale + input_zero_point

    predictions = output.argmax()

    return predictions

tflite_model_quant_file = "./quan_model/kws_quant.tflite"
predict_result = predict_quan_model(tflite_model_quant_file, input_wav)
print("The predict lable is %s"%predict_result)