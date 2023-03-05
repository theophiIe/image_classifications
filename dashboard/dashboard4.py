import dash_bootstrap_components as dbc
import pandas as pd
import sys

from dash import html

sys.path.append('../')


def serve_layout():
    data = pd.read_json("../info_models.json")

    Card_1 = dbc.Card([

    ])

    Card_2 = dbc.Card([

    ])

    Card_3 = dbc.Card([

    ])

    Card_4 = dbc.Card([

    ])

    layout = html.Div([
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        Card_1,
                        html.Br(),
                        Card_3
                    ]
                ),
                dbc.Col(
                    [
                        Card_2,
                        html.Br(),
                        Card_4
                    ]
                ),
            ],
            justify="center",
        ),
    ])

    return layout
