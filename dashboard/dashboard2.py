import dash_bootstrap_components as dbc
import glob
import pandas as pd
import plotly.express as px
import re

from dash import Input, Output, html, callback, dcc


def serve_layout():
    df_accuracy = pd.read_csv('./accuracy/accuracy_1.csv', index_col=[0])
    df_cosine = pd.read_csv('./cosine/cosine_1.csv', index_col=[0])

    fig1 = px.line(df_accuracy, x="limit", y="score")
    fig2 = px.imshow(df_cosine, text_auto=True, aspect="auto")

    Card_1 = dbc.Card([
        dbc.CardHeader("Card_1", style={"text-align": "center"}),
        dcc.Graph(id="plot", figure=fig1),
        html.Br()
    ])

    Card_2 = dbc.Card([
        dbc.CardHeader("Card_2", style={"text-align": "center"}),
        dcc.Graph(id="matrix", figure=fig2),
        html.Br()
    ])

    models = sorted(glob.glob("./vectors/*.npy"))
    model_choice = dcc.Dropdown([m for m in models], "vectors.npy",
                                id='model-choice',
                                style={'font-size': 20, 'width': '50%', 'align': 'center'})

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
    index = re.search(r'[0-9]+', value).group() if value is not None and value != "" else 1

    df_accuracy = pd.read_csv(f'./accuracy/accuracy_{index}.csv', index_col=[0])
    df_cosine = pd.read_csv(f'./cosine/cosine_{index}.csv', index_col=[0])

    fig1 = px.line(df_accuracy, x="limit", y="score")
    fig2 = px.imshow(df_cosine, text_auto=True, aspect="auto")

    return fig1, fig2
