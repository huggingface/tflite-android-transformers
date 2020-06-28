import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering

model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

input_spec = tf.TensorSpec([1, 384], tf.int32)
model._set_inputs(input_spec, training=False)


# For tensorflow>2.2.0, set inputs in the following way.
# Otherwise, the model.inputs and model.outputs will be None.
# keras_input = tf.keras.Input([384], batch_size=1, dtype=tf.int32)
# keras_output = model(keras_input, training=False)
# model = tf.keras.Model(keras_input, keras_output)


print(model.inputs)
print(model.outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For normal conversion:
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]

# For conversion with FP16 quantization:
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_types = [tf.float16]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_new_converter = True

# For conversion with hybrid quantization:
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.experimental_new_converter = True

tflite_model = converter.convert()

open("distilbert-squad-384.tflite", "wb").write(tflite_model)
