import glob
import math
import re
import numpy as np
import pandas as pd
from PIL import Image

import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
from dash import Input, Output, State, dcc, html, dash_table, callback, callback_context, dcc
from dash.exceptions import PreventUpdate
from generate_graphs import find_clusters, generate_main_graph, generate_cluster_graph

vectors = np.load('vectors.npy')
names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('./img/*'))))
clusters = find_clusters(vectors, names, 'cosine', 0.2)

def serve_layout():
    
    Card_1= dbc.Card([
        dbc.CardHeader("Main graph", style={"text-align": "center"}),
        html.Div([
            cyto.Cytoscape(
                id='main_graph',
                layout={'name': 'circle'},
                style={'width': '100%', 'height': '600px'},
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

    Card_2= dbc.Card([
        dbc.CardHeader("Card_2", id="card-2", style={"text-align": "center"}),
        html.Div([
            cyto.Cytoscape(
                id='clusters_graph',
                layout={'name': 'circle'},
                style={'width': '100%', 'height': '600px'},
                elements=generate_cluster_graph(clusters, "Humain")
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
    ])

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

    layout = html.Div([
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
                        Card_2
                    ]
                ),
            ],
            justify="center"
        ),
    ])

    return layout

@callback(Output('clusters_graph', 'elements'),
          Output('card-2', 'children'),
          Input('main_graph','tapNode'))
def update_cluster(node):
    if node != None:
        cluster_type = re.search(r'[a-zA-Z]+', str(node['data']['id'])).group()
        return generate_cluster_graph(clusters, cluster_type), cluster_type
    return generate_cluster_graph(clusters, "Humain"), "Humain"

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
        if node2 != None:
            if node2['classes'] == 'img_node':
                image = Image.open("./img/"+node2['data']['id']+".jpg")
                return True, html.Img(src=image, width='100%',  alt='image')
            return False, " "
    elif input_id == "main_graph":
        if node1 != None:
            if node1['classes'] == 'img_node':
                image = Image.open("./img/"+node1['data']['id']+".jpg")
                return True, html.Img(src=image, width='100%',  alt='image')
            return False, " "
    return False, " "