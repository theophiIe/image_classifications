import glob
import os

import numpy as np
import re
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image


def extract(model, file, size):
    file = Image.open(file).convert('L').resize(size)
    file = np.stack((file,) * 3, axis=-1)
    file = np.array(file) / 255.0

    embedding = model.predict(file[np.newaxis, ...])

    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()

    return flattended_feature


def vectorized_images(model, size, paths):
    vectorized_img = []

    for image in paths:
        vectorized_img.append(extract(model, image, size))

    return vectorized_img


if __name__ == '__main__':
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

    images_paths = sorted(glob.glob("./img/*"))
    images_names = list(map(lambda image_name: re.search(r'/([a-zA-Z0-9]+)\.', image_name).group()[1:-1], images_paths))

    extract_images_google_1 = vectorized_images(model_google_1, (299, 299), images_paths)
    extract_images_google_2 = vectorized_images(model_google_2, (224, 224), images_paths)
    extract_images_google_3 = vectorized_images(model_google_3, (224, 224), images_paths)
    extract_tensorflow_1 = vectorized_images(model_tensorflow_1, (224, 224), images_paths)
    extract_images_google_4 = vectorized_images(model_google_4, (512, 512), images_paths)
    extract_images_google_5 = vectorized_images(model_google_5, (224, 224), images_paths)
    extract_images_google_6 = vectorized_images(model_google_6, (240, 240), images_paths)

    if not os.path.exists("./vectors"):
        os.mkdir("./vectors")

    else:
        files = glob.glob('./vectors/*')
        for f in files:
            os.remove(f)

    np.save('./vectors/vectors_google_1.npy', extract_images_google_1)
    np.save('./vectors/vectors_google_2.npy', extract_images_google_2)
    np.save('./vectors/vectors_google_3.npy', extract_images_google_3)
    np.save('./vectors/vectors_tensorflow_4.npy', extract_tensorflow_1)
    np.save('./vectors/vectors_google_4.npy', extract_images_google_4)
    np.save('./vectors/vectors_google_5.npy', extract_images_google_5)
    np.save('./vectors/vectors_google_6.npy', extract_images_google_6)
