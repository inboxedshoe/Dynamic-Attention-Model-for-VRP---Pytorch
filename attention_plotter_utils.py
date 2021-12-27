
import numpy as np

import plotly.graph_objs as go
import networkx as nx

#creates the data graph instance
def generate_graph(pts, instance=0, dictionary=True):
    if not dictionary:
        # temp demand list with depot and nodes
        temp_demand = pts[instance][2]
        temp_demand.insert(0, 0)

        # temp pos list with depot and nodes
        temp_pos = pts[instance][1]
        temp_pos.insert(0, pts[instance][0])

    else:
        # temp demand list with depot and nodes
        temp_demand = pts["demand"][instance].cpu().tolist()
        temp_demand.insert(0, 0)

        # temp demand list with depot and nodes
        temp_pos = pts["loc"][instance].cpu().tolist()
        temp_pos.insert(0, pts["depot"][instance].cpu().tolist())

    graph = nx.complete_graph(len(temp_pos))

    for i in range(0, len(temp_pos)):
        graph.add_node(i, pos=temp_pos[i], demand=temp_demand[i])

    # create fully_connected
    # graph = nx.complete_graph(graph)

    # set attributes
    # nx.set_node_attributes(graph, dict(enumerate(temp_demand) ), "demand")

    return graph


#this will create our initial plotly graph to display
def create_plotly(G, logs):

    #get the nodes and edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    #create a scatter plot
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    #place the nodes in the respective position in the plot
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    #customize the nodes (change color, shape....)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlOrRd',
            reversescale=False,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title='attention_prob',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_trace.marker.color = np.exp(logs[0])
    node_trace.text = np.exp(logs[0])

    node_trace.marker.line.color = "black"
    node_trace.marker.symbol = ["square"]

    # draw
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                    )

    fig.update_layout(
        width=800,
        height=800
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig
