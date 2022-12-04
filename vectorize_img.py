from dash import Dash, html
import dash_cytoscape as cyto
import glob
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
from scipy.spatial import distance

model_google_1 = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5", trainable=False, arguments=dict(batch_norm_momentum=0.997))])


def extract(model, file, size):
  file = Image.open(file).convert('L').resize(size)
  # display(file)

  file = np.stack((file,)*3, axis=-1)
  file = np.array(file) / 255.0

  embedding = model.predict(file[np.newaxis, ...])
  # print(embedding)
  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()

  # print(len(flattended_feature))
  # print(flattended_feature)
  # print('-----------')
  return flattended_feature

def vectorized_images(model, size, images_path):
    vectorized_img = []
    names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob(images_path))))
    for img in sorted(glob.glob(images_path)):
        vectorized_img.append(extract(model, img, size))
    return vectorized_img, names

if __name__ == '__main__':

    vectors, names = vectorized_images(model_google_1, (299,299), "./img/*")
    np.save('vectors.npy', vectors)