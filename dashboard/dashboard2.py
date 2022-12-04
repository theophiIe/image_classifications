import glob
import math
import re
import numpy as np
import pandas as pd
import plotly.express as px

import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import Input, Output, State, dcc, html, dash_table, callback, callback_context, dcc
from dash.exceptions import PreventUpdate
from line_charts import generate_data_frame, plot_accuracy

images_names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('./img/*'))))
models = sorted(glob.glob("*.npy"))
vectors = np.load('vectors.npy')
limite_google_1, df_google_1 = plot_accuracy(vectors, images_names)
df_cosine_google_1 = generate_data_frame(vectors, 'cosine', images_names)

def serve_layout():
    vectors = np.load('vectors.npy')
    #images_paths = sorted(glob.glob("./img/*"))
    limite_google_1, df_google_1 = plot_accuracy(vectors, images_names)
#
    #df_cosine_google_1 = generate_data_frame(vectors, 'cosine', images_names)
#
    df = df_google_1
    fig = px.line(df, x="limit", y="score")
    fig2 = px.imshow(df_cosine_google_1, text_auto=True, aspect="auto")

    Card_1= dbc.Card([
        dbc.CardHeader("Card_1", style={"text-align": "center"}),
        dcc.Graph(id="plot", figure=fig),
        html.Br()
    ])


    Card_2= dbc.Card([
        dbc.CardHeader("Card_2", style={"text-align": "center"}),
        dcc.Graph(id="matrix", figure=fig2),
        html.Br()
    ])

    model_choice = dcc.Dropdown([m for m in models], "vectors.npy",
                                id='model-choice', 
                                style={'font-size': 20, 'width':'50%', 'align': 'center'})


    layout = html.Div([
        dbc.Row(model_choice),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        Card_1
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

@callback(Output('plot', 'figure'),
          Output('matrix', 'figure'),
          Input('model-choice', 'value'))
def update_model(value):
    vectors = np.load(value)
    limite_google_1, plot  = plot_accuracy(vectors, images_names)
    df = generate_data_frame(vectors, 'cosine', images_names)
    fig = px.line(plot, x="limit", y="score")
    fig2 = px.imshow(df, text_auto=True, aspect="auto")

    return fig, fig2