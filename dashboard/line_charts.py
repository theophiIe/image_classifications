import glob
import re
import statistics

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from scipy.spatial import distance


def find_clusters(vectors, metric, limit, names):
    clusters = {}
    for index1, v1 in enumerate(vectors):
        cluster_type = re.search(r'[a-zA-Z]+', str(names[index1])).group()
        if not cluster_type in clusters.keys():
            clusters[cluster_type] = []
        for index2, v2 in enumerate(vectors):
            d = distance.cdist([v1], [v2], metric)[0]
            if limit > d[0] > 0.0001:
                if names[index2] not in clusters[cluster_type]:
                    clusters[cluster_type].append(names[index2])
    return clusters

def get_precision(cluster):
    l = []
    for key, value in cluster.items():
        nb_correct = 0
        for v in value:
            if re.search(r'[a-zA-Z]+', v).group() == key:
                nb_correct += 1
        l.append((nb_correct / len(value)) if len(value) != 0 else 0)
    return l

def get_recall(cluster, names):
    l = []
    for key, value in cluster.items():
        nb_correct = 0
        for v in value:
            if re.search(r'[a-zA-Z]+', v).group() == key:
                nb_correct += 1
        len_list = [re.search(r'[a-zA-Z]+', i).group() for i in names if re.search(r'[a-zA-Z]+', i).group() == key]
        l.append((nb_correct / len(len_list)) if len(len_list) != 0 else 0)
    return l

def get_f1_score(precision_scores, recall_scores):
    l = []
    for p, r in zip(precision_scores, recall_scores):
        l.append((2 * ((p * r) / (p + r))) if p + r != 0 else 0)
    return l

def get_accuracy(f1_scores):
    return statistics.mean(f1_scores)

def plot_accuracy(extract_images, names):
    score = []
    optimal_accuracy, optimal_limite = 0, 0

    for limit in np.arange(0.1, 0.95, 0.05):
        clusters = find_clusters(extract_images, 'cosine', limit, names)
        precision = get_precision(clusters)
        recall = get_recall(clusters, names)
        f1_score = get_f1_score(precision, recall)
        accuracy = get_accuracy(f1_score)
        optimal_accuracy, optimal_limite = (accuracy, limit) if optimal_accuracy < accuracy else (optimal_accuracy, optimal_limite)
        score.append(accuracy)

    df_accuracy = pd.DataFrame({'score': score, 'limit': np.arange(0.1, 0.95, 0.05)})

    return optimal_limite, df_accuracy

def generate_data_frame(extract_images, metric, name_images):
    dict_res = {}

    for index, image_1 in enumerate(extract_images):
        res = []
        for image_2 in extract_images:
            dc = distance.cdist([image_1], [image_2], metric)[0]
            res.append(dc[0])

        dict_res[name_images[index]] = res

    dataframe = pd.DataFrame(data=dict_res)
    dataframe.index = name_images

    return dataframe

