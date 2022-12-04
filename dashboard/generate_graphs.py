import re
import tensorflow as tf
import tensorflow_hub as hub

from scipy.spatial import distance

model_google_1 = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
                                                     trainable=False, arguments=dict(batch_norm_momentum=0.997))])


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
                if names[index2] not in clusters[cluster_type]:
                    clusters[cluster_type].append({names[index2]: v2})
                    clustered_items.append(names[index2])
    for index, v in enumerate(vectors):
        if (names[index] not in clustered_items) and (names[index] not in clusters["NoCluster"]):
            clusters["NoCluster"].append({names[index]: v})
    return clusters


def generate_main_graph(cluster):
    cyto_nodes = []
    colors = ['red', 'blue', 'green', 'yellow', 'violet']
    for key, values in cluster.items():
        indice_key = list(cluster.keys()).index(key)
        cyto_nodes.append({'data': {'id': key, 'label': key.capitalize()},
                           'classes': 'cluster_node',
                           'style': {'background-color': colors[indice_key], 'width': 100, 'height': 90,
                                     'font-size': 150},
                           'selectable': True
                           })
        for d in values:
            for k, v in d.items():
                cyto_nodes.append({'data': {'id': k, 'label': k},
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


def generate_cluster_graph(cluster, cluster_type):
    cyto_nodes = []
    colors = ['red', 'blue', 'green', 'yellow', 'violet']
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
                            if 0.2 > d[0] > 0.0001:
                                cyto_edges.append({'data': {'source': k, 'target': k2},
                                                   'classes': 'edge',
                                                   'style': {'line-color': colors[indice_key]}})

    return cyto_nodes + cyto_edges
