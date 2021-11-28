import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention
import math

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Module):
    """Feed-Forward Sublayer: fully-connected Feed-Forward network,
    built based on MHA vectors from MultiHeadAttention layer with skip-connections

        Args:
            num_heads: number of attention heads in MHA layers.
            input_dim: embedding size that will be used as d_model in MHA layers.
            feed_forward_hidden: number of neuron units in each FF layer.

        Call arguments:
            x: batch of shape (batch_size, n_nodes, node_embedding_size).
            mask: mask for MHA layer

        Returns:
               outputs of shape (batch_size, n_nodes, input_dim)

    """

    def __init__(self, input_dim, num_heads, feed_forward_hidden=512, attention_type=0, batch_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(n_heads=num_heads, d_model=input_dim, attention_type=attention_type)
        
        self.ff1 = nn.Linear(input_dim, feed_forward_hidden)
        self.ff2 = nn.Linear(feed_forward_hidden, input_dim)

        self.batch_norm = batch_norm
        if self.batch_norm:
            # self.norm_MHA = get_norm(norm_type, hdim=embed_dim, **kwargs)
            # self.norm_linear = get_norm(norm_type, hdim=embed_dim, **kwargs)
            self.norm_MHA = Normalization(input_dim, "batch")
            self.norm_linear = Normalization(input_dim, "batch")

    def forward(self, x, mask=None):
        mha_out = self.mha(x, x, x, mask)
        sc1_out = torch.add(x, mha_out)
        tanh1_out = torch.tanh(sc1_out)
        if self.batch_norm:
            tanh1_out = self.norm_MHA(tanh1_out)

        ff1_out = self.ff1(tanh1_out)
        relu1_out = F.relu(ff1_out)
        ff2_out = self.ff2(relu1_out)
        sc2_out = torch.add(tanh1_out, ff2_out)
        tanh2_out = torch.tanh(sc2_out)
        if self.batch_norm:
            tanh2_out = self.norm_linear(tanh2_out)

        return tanh2_out

class GraphAttentionEncoder(nn.Module):
    """Graph Encoder, which uses MultiHeadAttentionLayer sublayer.

        Args:
            input_dim: embedding size that will be used as d_model in MHA layers.
            num_heads: number of attention heads in MHA layers.
            num_layers: number of attention layers that will be used in encoder.
            feed_forward_hidden: number of neuron units in each FF layer.

        Call arguments:
            x: tuples of 3 tensors:  (batch_size, 2), (batch_size, n_nodes-1, 2), (batch_size, n_nodes-1)
            First tensor contains coordinates for depot, second one is for coordinates of other nodes,
            Last tensor is for normalized demands for nodes except depot

            mask: mask for MHA layer

        Returns:
               Embedding for all nodes + mean embedding for graph.
               Tuples ((batch_size, n_nodes, input_dim), (batch_size, input_dim))
    """

    def __init__(self, input_dim, num_heads, num_layers, feed_forward_hidden=512, attention_type=0, batch_norm=False):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feed_forward_hidden = feed_forward_hidden

        # initial embeddings (batch_size, n_nodes-1, 2) --> (batch-size, input_dim), separate for depot and other nodes
        self.init_embed_depot = nn.Linear(2, self.input_dim)  # nn.Linear(2, embedding_dim)
        self.init_embed = nn.Linear(3, self.input_dim)

        self.mha_layers = [MultiHeadAttentionLayer(self.input_dim, self.num_heads, self.feed_forward_hidden, attention_type=attention_type, batch_norm=batch_norm)
                            for _ in range(self.num_layers)]
        self.mha_layers = nn.ModuleList(self.mha_layers)

    def forward(self, x, mask=None, cur_num_nodes=None):

        x = torch.cat((self.init_embed_depot(x[0])[:, None, :],  # (batch_size, 2) --> (batch_size, 1, 2)
                       self.init_embed(torch.cat((x[1], x[2][:, :, None]), -1))  # (batch_size, n_nodes-1, 2) + (batch_size, n_nodes-1)
                       ), 1)  # (batch_size, n_nodes, input_dim)

        # stack attention layers
        for i in range(self.num_layers):
            x = self.mha_layers[i](x, mask)

        if mask is not None:
            output = (x, torch.sum(x, 1) / cur_num_nodes)
        else:
            output = (x, torch.mean(x, 1))
            
        return output # (embeds of nodes, avg graph embed)=((batch_size, n_nodes, input), (batch_size, input_dim))