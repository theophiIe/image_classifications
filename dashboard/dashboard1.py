import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import glob
import numpy as np
import re
import pandas as pd

from dash import Input, Output, html, callback, callback_context, dcc, State
from generate_graphs import find_clusters, generate_main_graph, generate_cluster_graph
from dash_matrice import nb_pred
from PIL import Image

vectors = np.load('../vectors/vectors_google_1.npy')
names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
clusters = find_clusters(vectors, names, 'cosine', 0.3)
data = pd.read_json("../info_models.json")


def serve_layout():
    limit = dcc.Slider(0, 0.7, 0.05,
                       value=0.3,
                       id='limit-slider'
                       )
    models = sorted(glob.glob("../vectors/*.npy"))
    model_choice = dcc.Dropdown([m for m in models], "../vectors/vectors_google_1.npy",
                                id='model-choice-1',
                                style={'font-size': 15, 'width': '50%'})
    cluster_limit = dcc.Slider(0, 0.7, 0.05,
                               value=0.2,
                               id='cluster-limit-slider'
                               )

    Card_1 = dbc.Card([
        dbc.CardHeader("Main graph", style={"text-align": "center"}),
        dbc.Row(limit),
        html.Div([
            cyto.Cytoscape(
                id='main_graph',
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '500px'},
                elements=generate_main_graph(clusters)
            )
        ]),
        dcc.Dropdown(
            id='dropdown-update-layout',
            value='cose',
            clearable=False,
            options=[
                {'label': name.capitalize(), 'value': name}
                for name in ['grid', 'random', 'circle', 'cose', 'concentric']
            ]
        ),
    ])

    about_window = dbc.Col([
        dbc.Button("?", id="open-about", n_clicks=0),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("About")),
            dbc.Card([
                dbc.CardHeader(id="about-title", style={"text-align": "center"}),
                dbc.CardBody(
                    [
                        html.H5(id="about-name", className="card-title"),
                        html.P(id="about-description"),
                        html.P(id="about-publisher"),
                        html.P(id="about-imgsize"),
                        html.P(id="about-vectorsize"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button(id="link", outline=True, color="primary", target="_blank"),
                                dbc.Button(id="Architecture", outline=True, color="primary",
                                           target="_blank"),
                                dbc.Button(id="Dataset", outline=True, color="primary", target="_blank"),
                            ]
                        ),
                    ]
                )
            ]),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id="close-about", className="ms-auto", n_clicks=0
                )
            ),
        ],
            id="modal-about",
            is_open=False,
        ),
    ],
        width=1
    )

    Tab_1 = dbc.Tab([
        html.Br(),
        dbc.Row(cluster_limit),
        html.Div([
            cyto.Cytoscape(
                id='clusters_graph',
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '465px'},
                elements=generate_cluster_graph(clusters, "Humain", 0.2)
            )
        ]),
        dcc.Dropdown(
            id='dropdown-update-layout-cluster',
            value='cose',
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
        dbc.Row([model_choice, dbc.Col(id="sizes", width=5), about_window]),
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
          Output("sizes", "children"),
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
                                      limit), cluster_type, f"Confusion matrix for {cluster_type}", dbc.Col(
            [html.P("Image size = " + data[value]["image_size"] + ", Vector size = " + data[value]["vector_size"])])

    return generate_cluster_graph(clusters, "Humain", limit), "Humain", "Confusion matrix for Humain", dbc.Col(
        [html.P("Image size = " + data[value]["image_size"] + ", Vector size = " + data[value]["vector_size"])])


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


@callback(
    Output("modal-about", "is_open"),
    [Input("open-about", "n_clicks"), Input("close-about", "n_clicks")],
    [State("modal-about", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(Output("about-title", "children"),
          Output("about-name", "children"),
          Output("about-description", "children"),
          Output("about-publisher", "children"),
          Output("about-imgsize", "children"),
          Output("about-vectorsize", "children"),
          Output("link", "href"),
          Output("link", "children"),
          Output("Architecture", "href"),
          Output("Architecture", "children"),
          Output("Dataset", "href"),
          Output("Dataset", "children"),
          Input("model-choice-1", "value"))
def update_about(model):
    return data[model]["name"], "Name : " + data[model]["name"], "Description : " + data[model][
        "description"], "Publisher : " + data[model]["publisher"], "Image size : " + data[model][
                                    "image_size"], "Vector size : " + data[model]["vector_size"], data[model][
        "link"], "link", data[model]["architecture"], "architechture", data[model]["dataset"], "dataset"
