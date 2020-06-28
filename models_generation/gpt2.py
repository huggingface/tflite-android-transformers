import tensorflow as tf
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel.from_pretrained('gpt2') # or 'distilgpt2'

input_spec = tf.TensorSpec([1, 64], tf.int32)
model._set_inputs(input_spec, training=False)

# For tensorflow>2.2.0, set inputs in the following way.
# Otherwise, the model.inputs and model.outputs will be None.
# keras_input = tf.keras.Input([64], batch_size=1, dtype=tf.int32)
# keras_output = model(keras_input, training=False)
# model = tf.keras.Model(keras_input, keras_output)

print(model.inputs)
print(model.outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For FP16 quantization:
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

open("gpt2-64.tflite", "wb").write(tflite_model)
