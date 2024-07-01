import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualisation:

    @staticmethod
    def visualise_metrics(epochs: list, metrics: list, plot_title='', y_title=''):
        text = [y_title] * len(epochs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs,
                                 y=metrics,
                                 mode='lines+markers',
                                 hovertext=text,
                                 name='metrics'))
        fig.update_layout(title=plot_title,
                          xaxis_title='Numbers of training iteration',
                          yaxis_title=y_title)
        fig.update_traces(hoverinfo='all',
                          hovertemplate='Epoch: %{x}<br>%{hovertext}: %{y}')
        fig.show()

    @staticmethod
    def visualise_images(images: list[np.ndarray], predictions: list, plot_title=''):
        fig = make_subplots(rows=1, cols=len(images))
        for i in range(len(images)):
            images[i] = np.flip(np.reshape(images[i], (8, 8)), axis=0)
            fig.add_trace(go.Heatmap(z=images[i], name=str(predictions[i]), coloraxis='coloraxis'), row=1, col=i + 1)
        fig.update_layout(title=plot_title, coloraxis={'colorscale': 'gray_r'}, width=800, height=450)
        fig.show()