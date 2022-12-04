import math
import pandas as pd

import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import Input, Output, State, dcc, html, dash_table, callback, callback_context, dcc
from dash.exceptions import PreventUpdate

def serve_layout():

    Card_1= dbc.Card([
        dbc.CardHeader("Card_1", style={"text-align": "center"}),
        html.Br()
    ])

    buttons_card_1 = dbc.ButtonGroup(
        [dbc.Button("Left", outline=True, color="info"), dbc.Button("Right", outline=True, color="danger")],
        className="btn form-control",
    )

    Card_2= dbc.Card([
        dbc.CardHeader("Card_2", style={"text-align": "center"}),
        html.Br()
    ])

    buttons_card_2 = dbc.ButtonGroup(
        [dbc.Button("Left", outline=True, color="info"), dbc.Button("Right", outline=True, color="danger")],
        className="btn form-control",
    )

    layout = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        Card_1,
                        buttons_card_1
                    ]
                ),
                dbc.Col(
                    [
                        Card_2,
                        buttons_card_2
                    ]
                ),
            ],
            justify="center"
        ),
    ])

    return layout