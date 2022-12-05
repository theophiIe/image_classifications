import collections
import dash_bootstrap_components as dbc
import glob
import pandas as pd
import plotly.express as px
import re

from dash import Input, Output, html, callback, dcc


def serve_layout():
    df_accuracy = pd.read_csv('../accuracy/accuracy_1.csv', index_col=[0])
    df_cosine = pd.read_csv('../cosine/cosine_1.csv', index_col=[0])

    fig1 = px.line(df_accuracy, x="limit", y="score")
    fig2 = px.imshow(df_cosine, text_auto=True, aspect="auto")

    models = sorted(glob.glob("../vectors/*.npy"))
    models_choice = dcc.Dropdown([m for m in models], "../vectors/vectors_google_1.npy",
                                id='models-choice',
                                multi=True,
                                style={'font-size': 15, 'width': '100%'})

    model_choice = dcc.Dropdown([m for m in models], "../vectors/vectors_google_1.npy",
                                id='model-choice',
                                style={'font-size': 15, 'width': '100%'})
            

    Card_1 = dbc.Card([
        dbc.CardHeader("Card_1", style={"text-align": "center"}),
        dbc.Row(models_choice),
        dcc.Graph(id="plot", figure=fig1),
        html.Br()
    ])

    Card_2 = dbc.Card([
        dbc.CardHeader("Card_2", style={"text-align": "center"}),
        dbc.Row(model_choice),
        dcc.Graph(id="matrix", figure=fig2),
        html.Br()
    ])


    layout = html.Div([
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
          Input('models-choice', 'value'))
def update_model(value):
    colors = ['orange', 'magenta', 'green', 'red', 'blue']
    if type(value) is list:
        if value != []:
            index = [re.search(r'[0-9]+', v).group() for v in value]
        else:
            index = ['1']
    else:
        index = re.search(r'[0-9]+', value).group() if value is not None else 1

    dfs = [pd.read_csv(f'../accuracy/accuracy_{i}.csv', index_col=[0]) for i in index]

    fig = px.line(dfs[0], x="limit", y="score")
    for i in range(len(dfs)):
        fig.add_scatter(x = dfs[i].limit, y = dfs[i].score, line_color = colors[i], name = index[i])

    return fig

@callback(Output('matrix', 'figure'),
          Input('model-choice', 'value'))
def update_model(value):
    index = re.search(r'[0-9]+', value).group() if value is not None else 1
    df_cosine = pd.read_csv(f'../cosine/cosine_{index}.csv', index_col=[0])

    fig = px.imshow(df_cosine, text_auto=True, aspect="auto")

    return fig