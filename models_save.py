import tensorflow as tf
import tensorflow_hub as hub

model_google_1 = tf.keras.Sequential(
    [hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
                    trainable=False, arguments=dict(batch_norm_momentum=0.997))])

model_google_2 = tf.keras.Sequential([hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2", trainable=False)])

model_google_3 = tf.keras.Sequential([hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5", trainable=False,
    arguments=dict(batch_norm_momentum=0.997))])

model_tensorflow_1 = tf.keras.Sequential(
    [hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2",
                    input_shape=(224, 224) + (3,))])

model_google_4 = tf.keras.Sequential([hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2", trainable=False)])

model_google_5 = tf.keras.Sequential([hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2", trainable=False)])

model_google_6 = tf.keras.Sequential([hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2", trainable=False)])

tf.saved_model.save(model_google_1, 'saved_models/model_google_1')
tf.saved_model.save(model_google_2, 'saved_models/model_google_2')
tf.saved_model.save(model_google_3, 'saved_models/model_google_3')
tf.saved_model.save(model_tensorflow_1, 'saved_models/model_tensorflow_4')
tf.saved_model.save(model_google_4, 'saved_models/model_google_5')
tf.saved_model.save(model_google_5, 'saved_models/model_google_6')
tf.saved_model.save(model_google_6, 'saved_models/model_google_7')
