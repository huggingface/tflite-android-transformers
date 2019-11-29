import tensorflow as tf
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel.from_pretrained('gpt2') # or 'distilgpt2'

input_spec = tf.TensorSpec([1, 64], tf.int32)
model._set_inputs(input_spec, training=False)

print(model.inputs)
print(model.outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For FP16 quantization:
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

open("gpt2-64.tflite", "wb").write(tflite_model)
