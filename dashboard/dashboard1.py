import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import glob
import numpy as np
import re

from dash import Input, Output, html, callback, callback_context, dcc
from generate_graphs import find_clusters, generate_main_graph, generate_cluster_graph
from dash_matrice import nb_pred
from PIL import Image

vectors = np.load('../vectors/vectors_google_1.npy')
names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
clusters = find_clusters(vectors, names, 'cosine', 0.3)


def serve_layout():
    limit = dcc.Slider(0, 0.7, 0.05,
                       value=0.3,
                       id='limit-slider'
                       )
    models = sorted(glob.glob("../vectors/*.npy"))
    model_choice = dcc.Dropdown([m for m in models], "../vectors/vectors_google_1.npy",
                                id='model-choice-1',
                                style={'font-size': 15, 'width': '100%'})
    cluster_limit = dcc.Slider(0, 0.4, 0.05,
                               value=0.2,
                               id='cluster-limit-slider'
                               )

    Card_1 = dbc.Card([
        dbc.CardHeader("Main graph", style={"text-align": "center"}),
        dbc.Row(limit),
        html.Div([
            cyto.Cytoscape(
                id='main_graph',
                layout={'name': 'circle'},
                style={'width': '100%', 'height': '500px'},
                elements=generate_main_graph(clusters)
            )
        ]),
        dcc.Dropdown(
            id='dropdown-update-layout',
            value='circle',
            clearable=False,
            options=[
                {'label': name.capitalize(), 'value': name}
                for name in ['grid', 'random', 'circle', 'cose', 'concentric']
            ]
        ),
    ])

    Tab_1 = dbc.Tab([
        html.Br(),
        dbc.Row(cluster_limit),
        html.Div([
            cyto.Cytoscape(
                id='clusters_graph',
                layout={'name': 'circle'},
                style={'width': '100%', 'height': '465px'},
                elements=generate_cluster_graph(clusters, "Humain", 0.2)
            )
        ]),
        dcc.Dropdown(
            id='dropdown-update-layout-cluster',
            value='circle',
            clearable=False,
            options=[
                {'label': name.capitalize(), 'value': name}
                for name in ['grid', 'random', 'circle', 'cose', 'concentric']
            ]
        ),
    ], id="tab1")

    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(" ")),
                    dbc.ModalBody(" "),
                ],
                id="modal",
                is_open=False,
            ),
        ]
    )

    tabs = dbc.Card([
        dbc.CardHeader(
            dbc.Tabs([
                Tab_1,
                dbc.Tab(dcc.Graph(id="confusion-matrix", style={'height': '560px'}), id="tab2")
            ])
        )
    ])

    layout = html.Div([
        dbc.Row(model_choice),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        Card_1,
                        modal
                    ]
                ),
                dbc.Col(
                    [
                        tabs
                    ]
                ),
            ],
            justify="center",
        ),
    ])

    return layout


@callback(Output('clusters_graph', 'elements'),
          Output('tab1', 'label'),
          Output('tab2', 'label'),
          Input('main_graph', 'tapNode'),
          Input('model-choice-1', 'value'),
          Input('cluster-limit-slider', 'value'))
def update_cluster(node, value, limit):
    vectors = np.load(value)
    names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
    clusters = find_clusters(vectors, names, 'cosine', limit)
    if node is not None:
        cluster_type = re.search(r'[a-zA-Z]+', str(node['data']['id'])).group()
        return generate_cluster_graph(clusters, cluster_type,
                                      limit), cluster_type, f"Confusion matrix for {cluster_type}"

    return generate_cluster_graph(clusters, "Humain", limit), "Humain", "Confusion matrix for Humain"


@callback(Output('main_graph', 'layout'),
          Input('dropdown-update-layout', 'value'))
def update_layout(layout):
    return {
        'name': layout,
        'animate': True
    }


@callback(Output('clusters_graph', 'layout'),
          Input('dropdown-update-layout-cluster', 'value'))
def update_layout(layout):
    return {
        'name': layout,
        'animate': True
    }


@callback(
    Output("modal", "is_open"),
    Output("modal", "children"),
    Input('main_graph', 'tapNode'),
    Input('clusters_graph', 'tapNode')
)
def toggle_modal(node1, node2):
    input_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if input_id == "clusters_graph":
        if node2 is not None:
            if node2['classes'] == 'img_node':
                image = Image.open("../img/" + node2['data']['id'] + ".jpg")
                return True, html.Img(src=image, width='100%', alt='image')
            return False, " "
    elif input_id == "main_graph":
        if node1 is not None:
            if node1['classes'] == 'img_node':
                image = Image.open("../img/" + node1['data']['id'] + ".jpg")
                return True, html.Img(src=image, width='100%', alt='image')
            return False, " "
    return False, " "


@callback(Output('main_graph', 'elements'),
          Input('model-choice-1', 'value'),
          Input('limit-slider', 'value'))
def update_model_dash1(value, limit):
    vectors = np.load(value)
    names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
    clusters = find_clusters(vectors, names, 'cosine', limit)
    return generate_main_graph(clusters)


@callback(Output('confusion-matrix', 'figure'),
          Input('main_graph', 'tapNode'),
          Input('model-choice-1', 'value'),
          Input('cluster-limit-slider', 'value')
          )
def update_confusion_matrix(node, model, limit):
    colors = ['reds', 'blues', 'greens', 'oranges', 'magenta']
    vectors = np.load(model)
    names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
    clusters = find_clusters(vectors, names, 'cosine', limit)
    if node is not None:
        cluster_type = re.search(r'[a-zA-Z]+', str(node['data']['id'])).group()
    else:
        cluster_type = "Humain"
    index = list(clusters.keys()).index(cluster_type)
    return nb_pred(clusters, names, cluster_type, colors[index])
