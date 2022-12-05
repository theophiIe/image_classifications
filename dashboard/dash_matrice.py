import glob
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from scipy.spatial import distance
import statistics
import plotly.figure_factory as ff

def nb_pred(cluster, names, categorie, color):
    TP, TN, FN, FP = 0, 0, 0, 0
    for key, value in cluster.items():
        if(key == categorie):
            for d in value:
                for v in d.keys():
                    if re.search(r'[a-zA-Z]+', v).group() == categorie:
                        TP += 1
                    else:
                        FP += 1
            len_list = len([re.search(r'[a-zA-Z]+', i).group() for i in names if
                            re.search(r'[a-zA-Z]+', i).group() == categorie])
            FN = len_list - TP
            len_img = len(names)
            TN = len_img - len_list - FP
            cf_matrix = np.array([TP, FP, FN, TN])
            cf_matrix = cf_matrix.reshape(2, 2)
    return plot_confusion_matrix(cf_matrix, categorie, color)

def plot_confusion_matrix(cf_matrix, categorie, color):
    categories = [categorie, f'Non {categorie}']
    matrix = (cf_matrix / np.sum(cf_matrix))
    fig = px.imshow(matrix, x=categories, y=categories, text_auto=".2%", color_continuous_scale=color, aspect="auto")
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                  xaxis =  {"title": "Prédiction"}, 
                  yaxis = {"title": "Valeur actuelle"},
                  autosize = False,
                  width = 650, height = 550
                 )
    return fig
    

