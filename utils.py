import pickle
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time
import math

CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
}

def set_random_seed(seed):
    torch.manual_seed(seed)


def create_data_on_disk(graph_size, num_samples, is_save=True, filename=None, is_return=False, seed=1234):
    """Generate validation dataset (with SEED) and save
    """

    set_random_seed(seed)
    depo = torch.rand((num_samples, 2))

    set_random_seed(seed)
    graphs = torch.rand((num_samples, graph_size, 2))

    set_random_seed(seed)
    demand = torch.randint(low=1, high=10, size=(num_samples, graph_size), dtype=torch.float32) / CAPACITIES[graph_size]

    if is_save:
        save_to_pickle('./valsets/Validation_dataset_{}.pkl'.format(filename), (depo, graphs, demand))

    if is_return:
        return (depo, graphs, demand)


def save_to_pickle(filename, item):
    """Save to pickle
    """
    with open(filename, 'wb') as handle:
        pickle.dump(item, handle)


def read_from_pickle(path, return_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    objects = objects[0]
    if return_data_set:
        depo, graphs, demand = objects
        if num_samples is not None:
            return (depo[:num_samples], graphs[:num_samples], demand[:num_samples])
        else:
            return (depo, graphs, demand)
    else:
        return objects

def read_from_old_pickle(path, return_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    objects = tuple(objects[0])
    total_depo = []
    total_graphs = []
    total_demand = []

    for i in range(len(objects)):
        total_depo.append(objects[i][0])
        total_graphs.append(objects[i][1])
        total_demand.append([dem / objects[i][3] for dem in objects[i][2]])

    objects = (total_depo, total_graphs, total_demand)
    if return_data_set:
        if num_samples is not None:
            return (total_depo[:num_samples], total_graphs[:num_samples], total_demand[:num_samples])
        else:
            return (total_depo, total_graphs, total_demand)
    else:
        return objects


def generate_data_onfly(num_samples=10000, graph_size=20, dense_mix=1.0, extra_sizes=None):
    """Generate temp dataset in memory

        inputs:
        num_samples: total number of data samples to generate
        graph_size: largest graph size in the training set
        dense_mix: compresses half of the training data by a specific density factor
        extra_size: creates multiple data graph sizes with equal sample portions
        for the smaller graph sizes y
        currently a combination of dense mix and extra size is not implemented
    """
    if extra_sizes is None:
        if dense_mix == 1.0:
            depo = torch.rand((num_samples, 2))
            graphs = torch.rand((num_samples, graph_size, 2))
            demand = torch.randint(low=1, high=10, size=(num_samples, graph_size), dtype=torch.float32) / CAPACITIES[graph_size]
            return (depo, graphs, demand)
        else:
            # we want a mixture of data densities in the training data
            depo, graphs, demand = generate_default_density(graph_size, num_samples / 2, CAPACITIES)
            depo_temp, graphs_temp, demand_temp = generate_dense_data(graph_size, num_samples / 2, CAPACITIES, dense_mix, 100)

            depo = torch.cat([depo, depo_temp], dim=0)
            graphs = torch.cat([graphs, graphs_temp], dim=0)
            demand = torch.cat([demand, demand_temp], dim=0)
    else:
        num_graphs_sizes = len(extra_sizes) + 1
        num_samples_per_extras = math.floor(num_samples/num_graphs_sizes)
        num_samples_original = num_samples - len(extra_sizes)*num_samples_per_extras

        # the depo and demand tensors don't change
        depo = torch.rand((num_samples, 2))
        # demand tensor size is based on the biggest graph size since they wont be considered anyways
        demand = torch.randint(low=1, high=10, size=(num_samples, graph_size), dtype=torch.float32) / CAPACITIES[graph_size]

        # create the original data graphs first
        g_size = torch.tensor([graph_size])[None, None, :]
        g_size = g_size.repeat(num_samples_original, 1, 2)
        graphs = torch.rand((num_samples_original, graph_size, 2))
        graphs = torch.cat([g_size, graphs], dim=-2)

        for size in extra_sizes:

            # create the temporary subgraph
            graph_extra_temp = torch.rand((num_samples_per_extras, size, 2))

            # repeat the first element to fill the remaining required size
            repeat_vector = torch.ones(size, dtype=torch.int)
            repeat_vector[0] = graph_size - size + 1
            graph_extra_temp = torch.repeat_interleave(graph_extra_temp, repeat_vector, dim=-2)

            # add the graph size indicator
            g_size = torch.tensor([size])[None, None, :]
            g_size = g_size.repeat(num_samples_per_extras, 1, 2)
            graph_extra_temp = torch.cat([g_size, graph_extra_temp], dim=-2)

            #add to total graphs
            graphs = torch.cat([graphs, graph_extra_temp], dim=0)

    return(depo, graphs, demand)


def generate_default_density(size, samples, capacities):
    samples = int(samples)
    depo = torch.rand((samples, 2))
    graphs = torch.rand((samples, size, 2))
    demand = torch.randint(low=1, high=10, size=(samples, size), dtype=torch.float32) / CAPACITIES[size]

    return (depo, graphs, demand)


def generate_dense_data(size, samples, capacities, max_interval, num_distros=10, mixed=True):

    # first we need to sample multiple ranges
    start_points = []
    for i in range(num_distros):
        point = torch.FloatTensor(1).uniform_(0, 1-max_interval)
        # round and append
        # start_points.append(torch.round(point, decimals=3).item())
        start_points.append(point.item())

    if samples % num_distros == 0:
        mini_batch_size = int(samples / num_distros)
        # create initialmini batch


        depo = torch.FloatTensor(mini_batch_size,2).uniform_(start_points[0], start_points[0] + max_interval)
        graphs = torch.FloatTensor(mini_batch_size, size, 2).uniform_(start_points[0], start_points[0] + max_interval)
        demand = (torch.FloatTensor(mini_batch_size, size).uniform_(0, 9).int() + 1).float() / capacities[size]

        # loop over remaining minibatches
        for start_index in range(1, len(start_points)):

            depo = torch.cat([depo, torch.FloatTensor(mini_batch_size, 2).uniform_(start_points[start_index],
                                                           start_points[start_index] + max_interval)], dim=0)
            graphs = torch.cat([graphs, torch.FloatTensor(mini_batch_size, size, 2).uniform_(start_points[start_index],
                                                               start_points[start_index] + max_interval)], dim=0)
            demand = torch.cat([demand, (torch.FloatTensor(mini_batch_size, size).uniform_(0, 9).int() + 1).float() / capacities[size]], dim=0)

    else:
        print("sample size is not divisible by number of distros")
        assert (0)

    return (depo, graphs, demand)

def get_results(train_loss_results, train_cost_results, val_cost, save_results=True, filename=None, plots=True):

    epochs_num = len(train_loss_results)

    df_train = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                  'loss': np.array(train_loss_results),
                                  'cost': np.array(train_cost_results),
                                  })
    df_test = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                 'val_сost': np.array(val_cost)})
    if save_results:
        df_train.to_excel('train_results_{}.xlsx'.format(filename), index=False)
        df_test.to_excel('test_results_{}.xlsx'.format(filename), index=False)

    if plots:
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(x='epochs', y='loss', data=df_train, color='salmon', label='train loss')
        ax2 = ax.twinx()
        sns.lineplot(x='epochs', y='cost', data=df_train, color='cornflowerblue', label='train cost', ax=ax2)
        sns.lineplot(x='epochs', y='val_сost', data=df_test, palette='darkblue', label='val cost').set(ylabel='cost')
        ax.legend(loc=(0.75, 0.90), ncol=1)
        ax2.legend(loc=(0.75, 0.95), ncol=2)
        ax.grid(axis='x')
        ax2.grid(True)
        plt.savefig('learning_curve_plot_{}.jpg'.format(filename))
        plt.show()

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def get_journey(batch, pi, title, ind_in_batch=0):
    """Plots journey of agent

    Args:
        batch: dataset of graphs
        pi: paths of agent obtained from model
        ind_in_batch: index of graph in batch to be plotted
    """

    # Remove extra zeros
    pi_ = get_clean_path(pi[ind_in_batch].numpy())

    # Unpack variables
    depo_coord = batch[0][ind_in_batch].numpy()
    points_coords = batch[1][ind_in_batch].numpy()
    demands = batch[2][ind_in_batch].numpy()
    node_labels = ['(' + str(x[0]) + ', ' + x[1] + ')' for x in enumerate(demands.round(2).astype(str))]

    # Concatenate depot and points
    full_coords = np.concatenate((depo_coord.reshape(1, 2), points_coords))

    # Get list with agent loops in path
    list_of_paths = []
    cur_path = []
    for idx, node in enumerate(pi_):

        cur_path.append(node)

        if idx != 0 and node == 0:
            if cur_path[0] != 0:
                cur_path.insert(0, 0)
            list_of_paths.append(cur_path)
            cur_path = []

    list_of_path_traces = []
    for path_counter, path in enumerate(list_of_paths):
        coords = full_coords[[int(x) for x in path]]

        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        list_of_path_traces.append(go.Scatter(x=coords[:, 0],
                                              y=coords[:, 1],
                                              mode="markers+lines",
                                              name=f"path_{path_counter}, length={total_length:.2f}",
                                              opacity=1.0))

    trace_points = go.Scatter(x=points_coords[:, 0],
                              y=points_coords[:, 1],
                              mode='markers+text',
                              name='destinations',
                              text=node_labels,
                              textposition='top center',
                              marker=dict(size=7),
                              opacity=1.0
                              )

    trace_depo = go.Scatter(x=[depo_coord[0]],
                            y=[depo_coord[1]],
                            text=['1.0'], textposition='bottom center',
                            mode='markers+text',
                            marker=dict(size=15),
                            name='depot'
                            )

    layout = go.Layout(title='<b>Example: {}</b>'.format(title),
                       xaxis=dict(title='X coordinate'),
                       yaxis=dict(title='Y coordinate'),
                       showlegend=True,
                       width=1000,
                       height=1000,
                       template="plotly_white"
                       )

    data = [trace_points, trace_depo] + list_of_path_traces
    print('Current path: ', pi_)
    fig = go.Figure(data=data, layout=layout)
    fig.show()
    
def get_cur_time():
    """Returns local time as string
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def get_clean_path(arr):
    """Returns extra zeros from path.
       Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
    """

    p1, p2 = 0, 1
    output = []

    while p2 < len(arr):

        if arr[p1] != arr[p2]:
            output.append(arr[p1])
            if p2 == len(arr) - 1:
                output.append(arr[p2])

        p1 += 1
        p2 += 1

    if output[0] != 0:
        output.insert(0, 0.0)
    if output[-1] != 0:
        output.append(0.0)

    return output


def get_dev_of_mod(model):
    return next(model.parameters()).device


def _open_data(path):
    return open(path, 'rb')

def get_lhk_solved_data(path_instances, path_sols):
    """
    - instances[i][0] -> depot(x, y)
    - instances[i][1] -> nodes(x, y) * samples
    - instances[i][2] -> nodes(demand) * samples
    - instances[i][3] -> capacity (of vehicle) (should be the same for all in theory)

    - sols[0][i][0] -> cost
    - sols[0][i][1] -> path (doesn't include depot at the end)
    - sols[1] -> ?
    - sols[0][1][2] -> ?
    """

    with _open_data(path_instances) as f:
        instances_data = pickle.load(f) 
    with _open_data(path_sols) as f:
        sols_data = pickle.load(f) 

    capacity_denominator = CAPACITIES[len(instances_data[0][1])]

    depot_locs = (list(map(lambda x: x[0], instances_data))) # (samples, 2)
    nodes_locs = (list(map(lambda x: x[1], instances_data)))  # (samples, nodes, 2)
    nodes_demand = (list(map(lambda x: list(map(lambda d: d/capacity_denominator, x[2])), instances_data))) # (samples, nodes)
    instances = (depot_locs, nodes_locs, nodes_demand)

    path_indices = list(map(lambda x: x[1], sols_data[0])) # (samples, path_len)
    costs = list(map(lambda x: x[0], sols_data[0])) # (samples)

    capacities = (list(map(lambda x: x[3], instances_data))) # (samples)

    return instances, path_indices, costs, capacities