import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from config import cfg
from data_prepare import _parse_example
from data_prepare import gen_data_batch


def gen_test_data(file_pattern, num_samples):
  files = tf.data.Dataset.list_files(file_pattern)
  dataset = files.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(_parse_example)
  dataset = dataset.take(num_samples)
  return dataset

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):

    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    # Check if the output type is quantized, then rescale output data to float
    if output_details['dtype'] == np.uint8:
      output_scale, output_zero_point = output_details["quantization"]
      test_image = test_image.astype(np.float32)
      test_image = test_image / input_scale + input_zero_point

    predictions[i] = output.argmax()

    print("current test %d st sample" %i)
  return predictions

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global test_images
  global test_labels

  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)

  accuracy = (np.sum(test_labels == predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))

"""
Evaluate the quantized model
"""
test_images_list = []
test_labels_list = []
test_datasets = gen_test_data(cfg.val.dataset, cfg.val.num_samples)

for _, (images, labels) in enumerate(test_datasets):
  test_images_list.append(images)
  test_labels_list.append(labels)

test_images = np.array(test_images_list)
test_labels = np.array(test_labels_list)

tflite_model_quant_file = "./quan_model/kws_quant.tflite"
evaluate_model(tflite_model_quant_file, model_type="Quantized")

## Baseline model test
# model = tf.keras.models.load_model("kws.h5")
# val_dataset = gen_data_batch(cfg.val.dataset, cfg.batch_size, is_training=False)
# test_loss, test_acc = model.evaluate(val_dataset, steps=(cfg.val.num_samples // cfg.batch_size), verbose=2)
# print(test_acc)