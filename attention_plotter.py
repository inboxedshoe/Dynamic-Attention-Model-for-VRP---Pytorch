import dash
from dash import dcc
from dash import html

import plotly.graph_objs as go
import numpy as np

import pickle
from attention_plotter_utils import generate_graph, create_plotly

import torch

# initializing data and prepping first instances

dataset_name = "vrp20_validation_seed4321.pkl"

instance = 0

#load the graph instance
with open("data/CVRP/" + dataset_name, 'rb') as dataset_file:
    dataset = pickle.load(dataset_file)



# First Graph
with open('outputs/cvrp_20/run_20_encoder_record_20211118T010313/epoch-100-encoder-outputs.pkl', 'rb') as handle:
    b = torch.load(handle)

graphs = b[0]
logs = b[1][instance].cpu().numpy()
state = b[2][instance].cpu().numpy()

state = np.insert(state, 0, 0)

G = generate_graph(graphs, instance=instance, dictionary=True)
fig = create_plotly(G, logs)


# Second Graph
with open('outputs/run_20_encoder_record_20211118T010313/epoch-100-encoder-outputs.pkl', 'rb') as handle_1:
    b_1 = pickle.load(handle_1)

graphs_1 = b_1[0]
logs_1 = b_1[1][instance].cpu().numpy()
state_1 = b_1[2][instance].cpu().numpy()

state_1 = np.insert(state_1,0,0)

G_1 = generate_graph(graphs_1, instance = instance, dictionary = True)
fig_1 = create_plotly(G_1)


################################# SERVER CODE ############################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

fig.update_layout(
    # plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='small Change',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for your data.', style={
        'textAlign': 'left',
        'color': colors['text']
    }),
    dcc.Graph(
        id='attention-graph',
        figure=fig
    ),

    html.Div(id='slider-output-container', style={
        'textAlign': 'left',
        'color': colors['text']
    }),

    dcc.Slider(
        id='my-slider',
        min=0,
        max=logs.shape[0] - 1,
        step=1,
        value=1,
    ),

    dcc.Graph(
        id='sparse-graph',
        figure=fig_1
    ),

    html.Div(id='slider-output-container_1', style={
        'textAlign': 'left',
        'color': colors['text']
    }),

    dcc.Slider(
        id='my-slider_1',
        min=0,
        max=logs_1.shape[0] - 1,
        step=1,
        value=1,
    )

])


@app.callback(
    dash.dependencies.Output('attention-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value')],
    [dash.dependencies.State('attention-graph', 'figure')])
def update_figure(value, figure):
    opacity = np.ones(logs.shape[1])
    opacity[state[:value]] = 0.5

    marker_line_colors = np.repeat("black", logs.shape[1])
    marker_line_colors[state[value]] = "green"

    fig = go.Figure(figure)
    fig.update_traces(
        # update opacity based on visited
        marker_opacity=opacity,

        # update colors based on new log
        marker_color=np.exp(logs[value]),

        # update marker line color based on current node
        marker_line_color=marker_line_colors,

        selector=dict(mode='markers')
    )

    fig.update_traces(
        text=np.exp(logs[value])
    )

    return fig


@app.callback(
    dash.dependencies.Output('sparse-graph', 'figure'),
    [dash.dependencies.Input('my-slider_1', 'value')],
    [dash.dependencies.State('sparse-graph', 'figure')])
def update_figure_1(value, figure):
    opacity = np.ones(logs_1.shape[1])
    opacity[state_1[:value]] = 0.5

    marker_line_colors = np.repeat("black", logs_1.shape[1])
    marker_line_colors[state_1[value]] = "green"

    fig_1 = go.Figure(figure)
    fig_1.update_traces(
        # update opacity based on visited
        marker_opacity=opacity,

        # update colors based on new log
        marker_color=np.exp(logs_1[value]),

        # update marker line color based on current node
        marker_line_color=marker_line_colors,

        selector=dict(mode='markers')
    )

    fig_1.update_traces(
        text=np.exp(logs_1[value])
    )
    return fig_1


@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    # dash.dependencies.Output('attention-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):
    return 'Step: {}'.format(value)


@app.callback(
    dash.dependencies.Output('slider-output-container_1', 'children'),
    # dash.dependencies.Output('attention-graph', 'figure'),
    [dash.dependencies.Input('my-slider_1', 'value')])
def update_output_1(value):
    return 'Step: {}'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)

