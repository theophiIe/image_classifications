import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from dash import Input, Output, html, callback, dcc
from generate_graphs import find_clusters, find_near_imgs, generate_upload_graph
from PIL import Image
from vectorize_img import extract

vectors = np.load('../vectors/vectors_google_1.npy')
names = list(map(lambda image_name: image_name[6:-4], sorted(glob.glob('../img/*'))))
clusters = find_clusters(vectors, names, 'cosine', 0.3)


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
            multiple=False
        ),
    ])

    cyto_graph = html.Div([
        cyto.Cytoscape(
            id='cyto_upload',
            layout={'name': 'cose'},
            style={'width': '100%', 'height': '465px'}
        )
    ])

    models = sorted(glob.glob("../vectors/*.npy"))
    model_choice = dcc.Dropdown([m for m in models], "../vectors/vectors_google_1.npy",
                                id='model-choice-3',
                                style={'font-size': 15, 'width': '100%'})

    cluster_limit = dcc.Slider(0.1, 0.7, 0.05,
                               value=0.3,
                               id='limit-slider'
                               )

    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(" ")),
                    dbc.ModalBody(" "),
                ],
                id="img_modal",
                is_open=False,
            ),
        ]
    )

    layout = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        model_choice,
                        cluster_limit,
                        upload,
                        dcc.Loading(
                            id="loading-2",
                            children=[html.Div([html.Div(id="loading-output-1")])],
                            type="circle"
                        ),
                        cyto_graph
                    ]
                ),
                dbc.Col(
                    [
                        modal
                    ]
                ),
            ],
            justify="center"
        ),
    ])

    return layout


@callback(Output('cyto_upload', 'elements'),
          Output("loading-output-1", "children"),
          Input('upload-image', 'filename'),
          Input('model-choice-3', 'value'),
          Input('limit-slider', 'value'))
def update_output(file, model, limit):
    if file is not None:
        model_google_1 = tf.keras.Sequential(
            [hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
                            trainable=False, arguments=dict(batch_norm_momentum=0.997))])

        model_google_2 = tf.keras.Sequential([hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2", trainable=False)])

        model_google_3 = tf.keras.Sequential([hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5", trainable=False,
            arguments=dict(batch_norm_momentum=0.997))])

        model_tensorflow_1 = tf.keras.Sequential(
            [hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2",
                            input_shape=(224, 224) + (3,))])

        models = {'../vectors/vectors_google_1.npy': [model_google_1, (299, 299)],
                  '../vectors/vectors_google_2.npy': [model_google_2, (224, 224)],
                  '../vectors/vectors_google_3.npy': [model_google_3, (224, 224)],
                  '../vectors/vectors_tensorflow_4.npy': [model_tensorflow_1, (224, 224)]}

        path = '../uploads/' + str(file)
        vector = extract(models[str(model)][0], path, models[str(model)][1])
        imgs = find_near_imgs(vector, limit, 'cosine', str(model))

        return generate_upload_graph(file, imgs), ""
    return {}, ""


@callback(
    Output("img_modal", "is_open"),
    Output("img_modal", "children"),
    Input('cyto_upload', 'tapNode')
)
def toggle_modal(node):
    if node is not None:
        if node['classes'] == 'uploaded_img_node':
            image = Image.open("../uploads/" + node['data']['id'])
            return True, html.Img(src=image, width='100%', alt='image')
        image = Image.open("../img/" + node['data']['id'] + '.jpg')
        return True, html.Img(src=image, width='100%', alt='image')
    return False, " "
