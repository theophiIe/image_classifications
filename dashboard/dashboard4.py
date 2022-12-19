import dash_bootstrap_components as dbc
import pandas as pd
import sys

from dash import html

sys.path.append('../')


def serve_layout():
    data = pd.read_json("../info_models.json")

    Card_1 = dbc.Card([
        dbc.CardHeader(data.name[0], style={"text-align": "center"}),
        dbc.CardBody(
            [
                html.H5(data.name[0], className="card-title"),
                html.P(data.description[0]),
                html.P(f'Publisher : {data.publisher[0]}'),
                html.P(f'Image size : ({data.image_size[0]})'),
                html.P(f'Vector size : {data.vector_size[0]}'),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Link", outline=True, color="primary", href=data.link[0], target="_blank"),
                        dbc.Button("Architecture", outline=True, color="primary", href=data.architecture[0],
                                   target="_blank"),
                        dbc.Button("Dataset", outline=True, color="primary", href=data.dataset[0], target="_blank"),
                    ]
                ),
            ]
        )
    ])

    Card_2 = dbc.Card([
        dbc.CardHeader(data.name[1], style={"text-align": "center"}),
        dbc.CardBody(
            [
                html.H5(data.name[1], className="card-title"),
                html.P(data.description[1]),
                html.P(f'Publisher : {data.publisher[1]}'),
                html.P(f'Image size : ({data.image_size[1]})'),
                html.P(f'Vector size : {data.vector_size[1]}'),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Link", outline=True, color="primary", href=data.link[1], target="_blank"),
                        dbc.Button("Architecture", outline=True, color="primary", href=data.architecture[1],
                                   target="_blank"),
                        dbc.Button("Dataset", outline=True, color="primary", href=data.dataset[1], target="_blank"),
                    ]
                ),
            ]
        )
    ])

    Card_3 = dbc.Card([
        dbc.CardHeader(data.name[2], style={"text-align": "center"}),
        dbc.CardBody(
            [
                html.H5(data.name[2], className="card-title"),
                html.P(data.description[2]),
                html.P(f'Publisher : {data.publisher[2]}'),
                html.P(f'Image size : ({data.image_size[2]})'),
                html.P(f'Vector size : {data.vector_size[2]}'),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Link", outline=True, color="primary", href=data.link[2], target="_blank"),
                        dbc.Button("Architecture", outline=True, color="primary", href=data.architecture[2],
                                   target="_blank"),
                        dbc.Button("Dataset", outline=True, color="primary", href=data.dataset[2], target="_blank"),
                    ]
                ),
            ]
        )
    ])

    Card_4 = dbc.Card([
        dbc.CardHeader(data.name[3], style={"text-align": "center"}),
        dbc.CardBody(
            [
                html.H5(data.name[3], className="card-title"),
                html.P(data.description[3]),
                html.P(f'Publisher : {data.publisher[3]}'),
                html.P(f'Image size : ({data.image_size[3]})'),
                html.P(f'Vector size : {data.vector_size[3]}'),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Link", outline=True, color="primary", href=data.link[3], target="_blank"),
                        dbc.Button("Architecture", outline=True, color="primary", href=data.architecture[3],
                                   target="_blank"),
                        dbc.Button("Dataset", outline=True, color="primary", href=data.dataset[3], target="_blank"),
                    ]
                ),
            ]
        )
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
