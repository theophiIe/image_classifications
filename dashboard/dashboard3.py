import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import glob
import numpy as np
import re
import datetime

from dash import Dash
from dash.dependencies import Input, Output, State

from dash import Input, Output, html, callback, callback_context, dcc
from generate_graphs import find_clusters, generate_main_graph, generate_cluster_graph
from PIL import Image

vectors = np.load('../vectors/vectors_google_1.npy')
names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
clusters = find_clusters(vectors, names, 'cosine', 0.3)

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

def serve_layout():

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    upload = html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-image-upload'),
    ])


    layout = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        upload
                    ]
                ),
                dbc.Col(
                    [
                        #
                    ]
                ),
            ],
            justify="center"
        ),
    ])

    return layout


@callback(Output('output-image-upload', 'children'),
            Input('upload-image', 'contents'),
            State('upload-image', 'filename'),
            State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children