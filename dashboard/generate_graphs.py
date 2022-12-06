import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('../')

from scipy.spatial import distance


def find_clusters(vectors, names, metric, limit):
    clusters = {}
    clustered_items = []
    clusters["NoCluster"] = []
    for index1, v1 in enumerate(vectors):
        cluster_type = re.search(r'[a-zA-Z]+', str(names[index1])).group()
        if not cluster_type in clusters.keys():
            clusters[cluster_type] = []
        for index2, v2 in enumerate(vectors):
            d = distance.cdist([v1], [v2], metric)[0]
            if limit > d[0] > 0.0001:
                if names[index2] not in clustered_items:
                    clusters[cluster_type].append({names[index2]: v2})
                    clustered_items.append(names[index2])
    for index, v in enumerate(vectors):
        if (names[index] not in clustered_items) and (names[index] not in clusters["NoCluster"]):
            clusters["NoCluster"].append({names[index]: v})
    print(clusters)
    return clusters


def generate_main_graph(cluster):
    cyto_nodes = []
    colors = ['red', 'blue', 'green', 'orange', 'violet']
    for key, values in cluster.items():
        indice_key = list(cluster.keys()).index(key)
        cyto_nodes.append({'data': {'id': key, 'label': key.capitalize()},
                           'classes': 'cluster_node',
                           'style': {'background-color': colors[indice_key], 'width': 100, 'height': 90,
                                     'font-size': 80},
                           'selectable': True
                           })
        for d in values:
            for k, v in d.items():
                cyto_nodes.append({'data': {'id': k},
                                   'classes': 'img_node',
                                   'style': {'background-color': colors[indice_key], 'font-size': 50},
                                   'selectable': True})

    cyto_edges = []
    for key, values in cluster.items():
        indice_key = list(cluster.keys()).index(key)
        for d in values:
            for k, v in d.items():
                cyto_edges.append({'data': {'source': key, 'target': k},
                                   'classes': 'edge',
                                   'style': {'line-color': colors[indice_key]}})

    return cyto_nodes + cyto_edges


def generate_cluster_graph(cluster, cluster_type, limit):
    cyto_nodes = []
    colors = ['red', 'blue', 'green', 'orange', 'violet']
    for key, values in cluster.items():
        if key == cluster_type:
            indice_key = list(cluster.keys()).index(key)
            for d0 in values:
                for k0, v0 in d0.items():
                    cyto_nodes.append({'data': {'id': k0, 'label': k0},
                                       'classes': 'img_node',
                                       'style': {'background-color': colors[indice_key]
                                                 }})

    cyto_edges = []
    for key, values in cluster.items():
        if key == cluster_type:
            indice_key = list(cluster.keys()).index(key)
            for d in values:
                for k, v in d.items():
                    for d2 in values:
                        for k2, v2 in d2.items():
                            d = distance.cdist([v], [v2], 'cosine')[0]
                            if limit > d[0] > 0.0001:
                                cyto_edges.append({'data': {'source': k, 'target': k2},
                                                   'classes': 'edge',
                                                   'style': {'line-color': colors[indice_key]}})

    return cyto_nodes + cyto_edges


def find_near_imgs(vector, limit, metric, model):
    imgs = []
    names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
    vectors = np.load(model)
    for index, v in enumerate(vectors):
        d = distance.cdist([vector], [v], metric)[0]
        if limit > d[0] > 0.0001:
            imgs.append({names[index]:v})
    return imgs


def generate_upload_graph(uploaded_img, imgs):
    cyto_nodes = []
    cyto_edges = []
    cyto_nodes.append({'data': {'id': uploaded_img, 'label': uploaded_img},
                                'classes': 'uploaded_img_node'})
    colors = ['red', 'blue', 'green', 'orange', 'violet']
    for i in imgs:
        for key, value in i.items():
            cyto_nodes.append({'data': {'id': key, 'label': key},
                                'classes': 'img_node'})
            cyto_edges.append({'data': {'source': uploaded_img, 'target': key},
                                'classes': 'edge'})

    return cyto_nodes + cyto_edges