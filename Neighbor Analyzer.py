# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 22:10:25 2022

@author: inbox
"""
import torch

from attention_dynamic_model import set_decode_type
from utils import get_cur_time
from reinforce_baseline import load_pt_model
from utils import  read_from_old_pickle
from reinforce_baseline import validate

import tsplib95, requests
import lkh 

import numpy as np
import os

from more_itertools import split_at


def NormalizeData(data):
    if torch.min(data) == torch.max(data):
        return torch.ones_like(data)
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def flatten(t):
    return [item for sublist in t for item in sublist]


embedding_dim = 128
GRAPH_SIZE = 100
val_batch_size = 1000
normalize_cost = False
save_extras = False

#get model name and path
model_name = "mid/4/model/model_checkpoint_epoch_97_mixedNew_entmax_originalEncoder_50_2022-01-09"
MODEL_PATH = 'C:/Users/inbox/Desktop/Results/' + model_name

model = load_pt_model(MODEL_PATH,
                         embedding_dim=embedding_dim,
                         graph_size=GRAPH_SIZE,
                         attention_type=0,
                         attention_neighborhood=0,
                         batch_norm=False,
                         size_context=False,
                         normalize_cost=normalize_cost,
                         save_extras=save_extras,
                         device='cuda:0')

set_decode_type(model, "sampling")
print(get_cur_time(), 'model loaded')


#instances path
instance_path = 'C:/Users/inbox/Desktop/instances/100/'
files = os.listdir(instance_path)

vrps = []


for i in range(len(files)):
    if "sol" in files[i]:
        solution = []
        with open(instance_path + files[i], 'r') as solution_file:
            for line in solution_file:
                route = [int(s) for s in line.split() if s.isdigit()]
                if len(route) > 1: solution.append(route)
        problem = tsplib95.load(instance_path + files[i].replace(".sol",".vrp"))
        vrps.append({"name": files[i], "solution": solution, "coords":list(problem.node_coords.values()), "demands":list(problem.demands.values())})
        
    
# for vrp in vrps:
#     depot = torch.tensor([vrp["coords"][0:1]], dtype = torch.float32).reshape(1,2)
#     coords = torch.tensor([vrp["coords"][1:]], dtype = torch.float32)
#     demands = torch.tensor([vrp["demands"][1:]], dtype = torch.float32)

#     output = model([depot,coords,demands])       
        

route_neighbors = []
model_route_neighbors = []
for vrp in vrps:
    nodes = torch.tensor([vrp["coords"]], dtype = torch.float32)
    depot = NormalizeData(torch.tensor([vrp["coords"][0:1]], dtype = torch.float32).reshape(1,2))
    coords = NormalizeData(torch.tensor([vrp["coords"][1:]], dtype = torch.float32))
    demands = torch.tensor([vrp["demands"][1:]], dtype = torch.float32)/50
    
    # output = model([depot, coords, demands], return_pi=True)
    # temp_sol = output[2][0].tolist()
    # vrp["model_sol"] = list(split_at(temp_sol, lambda x: x == 0))[:][:-1]

    # get distance matrix
    distance_matrix = torch.cdist(nodes[0], nodes[0], p=2)
    # get location of nearest k items not including depot
    vals, idx = distance_matrix[:, :].topk(nodes[0].shape[0], largest=False)
    
    route_neighbor = []
    
    for route in vrp["solution"]:
        for i in range(len(route)):
            if i != len(route) - 1:
                route_neighbor.append((idx[route[i]] == route[i+1]).nonzero(as_tuple=True)[0].item())
    
    route_neighbors.append(route_neighbor)
    vrp["best_nn"] = route_neighbors
    
    # model_route_neighbor = []
    
    # for route in vrp["model_sol"]:
    #     for i in range(len(route)):
    #         if i != len(route) - 1:
    #             model_route_neighbor.append((idx[route[i]] == route[i+1]).nonzero(as_tuple=True)[0].item())
    
    # model_route_neighbors.append(model_route_neighbor)
    # vrp["model_nn"] = model_route_neighbors
            
    
    stats = []
    over_20 = 0
    
    for tour in route_neighbors:
        tour = np.array(tour)
        over_20 = over_20 + sum(tour>20)
        knn = {"average nn": np.around(tour.mean(),2),
               "std": np.around(np.std(tour),2),
               "max nn": tour.max(),
               "min nn": tour.min()}
        stats.append(knn)
        
    stats.append(over_20) 
    vrp["best_nn_stats"] = stats
    
    
    # stats = []
    # over_20 = 0
    
    # for tour in model_route_neighbors:
    #     tour = np.array(tour)
    #     over_20 = over_20 + sum(tour>20)
    #     knn = {"average nn": np.around(tour.mean(),2),
    #            "std": np.around(np.std(tour),2),
    #            "max nn": tour.max(),
    #            "min nn": tour.min()}
    #     stats.append(knn)
        
    # stats.append(over_20)   
    # vrp["model_nn_stats"] = stats
    
    
    
all_nn=[]
      
for vrp in vrps:
    for nn in vrp["best_nn"]:
        all_nn.append(nn)
        
import statistics as sc

all_nn = flatten(all_nn)
print("mean k = " + str(sc.mean(all_nn)))
print("std k = " + str(sc.stdev(all_nn)))
print("max k = " + str(max(all_nn)))
print("min k = " + str(min(all_nn)))
print("above 10 = " + str(sum(np.array(all_nn)>10)))

# model_all_nn=[]
      
# for vrp in vrps:
#     for nn in vrp["model_nn"]:
#         model_all_nn.append(nn)
        
# import statistics as sc

# model_all_nn = flatten(model_all_nn)
# print("model_mean k = " + str(sc.mean(model_all_nn)))
# print("model_std k = " + str(sc.stdev(model_all_nn)))
# print("model_max k = " + str(max(model_all_nn)))
# print("model_min k = " + str(min(model_all_nn)))


        
        
        
        
        
        