import tensorflow as tf
import pathlib
import sys
sys.path.append("..")
from config import cfg


train_tfrecord_file = cfg.train.dataset


# tf.train.Example decode
def _parse_example_wav(example_string):
    feature_description = {  # Define Feature structure
        'wave_data': tf.io.FixedLenFeature([], tf.string),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)

    wave_data = feature_dict['wave_data']
    wave_data = tf.io.decode_raw(wave_data, tf.int64)
    wave_data = tf.reshape(wave_data, [50, 12, 1])
    # Normalization of mfcc features
    wave_data = (wave_data - tf.math.reduce_min(wave_data)) / (tf.math.reduce_max(wave_data) - tf.math.reduce_min(wave_data))
    wave_data = tf.cast(wave_data, tf.float32)

    return wave_data

def gen_quan_data(file_pattern, num_samples):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_example_wav)
    dataset = dataset.batch(1)
    dataset = dataset.take(num_samples)
    return dataset

def representative_data_gen():
    for input_value in gen_quan_data(train_tfrecord_file, 1000):
        # Model has only one input so each data point has one element.
        yield [input_value]

model = tf.keras.models.load_model("kws.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model_quant = converter.convert()

# Save the quantized model
tflite_models_dir = pathlib.Path("./quan_model/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_quant_file = tflite_models_dir/"kws_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)