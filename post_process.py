# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 19:43:03 2022

@author: inbox
"""

from utils import read_from_old_pickle, FastTensorDataLoader,read_from_pickle
import numpy as np
import math
from tqdm import tqdm

import torch
from reinforce_baseline import load_pt_model
from attention_dynamic_model import set_decode_type
import os
import tsplib95, requests

CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.,
        199: 402.,
        1000: 131
}

def NormalizeData(data):
    if torch.min(data) == torch.max(data):
        return torch.ones_like(data)
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def normalize_problem(vrp):
    
    #nodes = torch.tensor([vrp["coords"]], dtype = torch.float32)
    depot = NormalizeData(torch.tensor([vrp["coords"][0:1]], dtype = torch.float32).reshape(1,2))
    coords = NormalizeData(torch.tensor([vrp["coords"][1:]], dtype = torch.float32))
    #demands = NormalizeData(torch.tensor([vrp["demands"][1:]], dtype = torch.float32))*9/CAPACITIES[coords.shape[1]]
    demands = torch.tensor([vrp["demands"][1:]], dtype = torch.float32)/CAPACITIES[coords.shape[1]]
    
    return (depot, coords, demands)

def load_instances(instance_path, dictionary = True):
    vrps = []
    
    files = os.listdir(instance_path)
    for i in range(len(files)):
        solution = []
        if "sol" in files[i]:
            with open(instance_path + files[i], 'r') as solution_file:
                for line in solution_file:
                    route = [int(s) for s in line.split() if s.isdigit()]
                    if len(route) > 1: solution.append(route)
            problem = tsplib95.load(instance_path + files[i].replace(".sol",".vrp"))
        else:
            problem = tsplib95.load(instance_path + files[i])
        vrps.append({"name": files[i], "solution": solution, "coords":list(problem.node_coords.values()), "demands":list(problem.demands.values())})

    return vrps

def rotate_single(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rotate_graph(coords_list, degrees = 0.5, origin = [0,0]):
    
    coords = np.array(coords_list)
    degrees = math.radians(degrees)
    
    coords = np.apply_along_axis(rotate_single, -1, coords, origin, degrees)
    
    return coords.tolist()

def dilate_single(coord, scale, origin):
    dilated = np.array([scale*(coord[0]-origin[0]) + origin[0],
                        scale*(coord[1]-origin[1]) + origin[1]])
    return dilated

def dilate_graph(coords_list, scale = 0.5, dilation_center = [0,0]):
    
    coords = np.array(coords_list)
    
    # coords[:,0] = scale*(coords[:,0]-dilation_center[0]) + dilation_center[0]
    # coords[:,1] = scale*(coords[:,1]-dilation_center[1]) + dilation_center[1]
    
    coords = np.apply_along_axis(dilate_single, -1, coords, scale, dilation_center)
    
    return coords.tolist()

def dilate_and_solve_instances(data_path, model_path, scales = [0.5], dilation_center = [0,0], rotations=[0], size = 100):
    
    embedding_dim = 128
    GRAPH_SIZE = size
    normalize_cost = False
    save_extras = False

    model = load_pt_model(model_path,
                         embedding_dim=embedding_dim,
                         graph_size=GRAPH_SIZE,
                         attention_type=0,
                         attention_neighborhood=0,
                         batch_norm=False,
                         size_context=False,
                         normalize_cost=normalize_cost,
                         save_extras=save_extras,
                         device='cuda:0')

    set_decode_type(model, "greedy")
    
    data = load_instances(data_path)
    
    for num, vrp in enumerate(data):
        
        data_normalized = normalize_problem(vrp)
        
        if num == 0:
            depots_normalized = data_normalized[0]
            coords_normalized = data_normalized[1]
            demands_normalized = data_normalized[2]
            
            depots_original = torch.tensor([vrp["coords"][0:1]], dtype = torch.float32).reshape(1,2)
            coords_original = torch.tensor([vrp["coords"][1:]], dtype = torch.float32)
            demands_original = (torch.tensor([vrp["demands"][1:]], dtype = torch.float32))
            
        else:     
            depots_normalized = torch.cat(depots_normalized, data_normalized[0], dim = 0)
            
            depots_original = torch.cat(depots_original, torch.tensor([vrp["coords"][0:1]], dtype = torch.float32).reshape(1,2), dim = 0)
            coords_original = torch.cat(coords_original,torch.tensor([vrp["coords"][1:]], dtype = torch.float32), dim = 0)
            demands_original =torch.cat(demands_original, (torch.tensor([vrp["demands"][1:]], dtype = torch.float32)))
            
                    
    data = [depots_original, coords_original, demands_original]
    data_normalized = [depots_normalized, coords_normalized, demands_normalized]

    model_was_training = model.training
    model.eval()
    
    #data = [torch.tensor(element) for element in data]

    with torch.no_grad():
        for i, scale in enumerate(scales):
            
            if i == 0:
                edited_data = data_normalized
                edited_data_temp = data_normalized
            else:
                edited_data = (dilate_graph(data_normalized[0], scale, dilation_center),
                                dilate_graph(data_normalized[1], scale, dilation_center),
                                data_normalized[2])
                edited_data_temp = edited_data
                
                
            for index, rotation in enumerate(rotations):
                
                sols_list = []
                costs_list = []
                
                edited_data = (rotate_graph(edited_data_temp[0], rotation, [0.5,0.5]),
                                rotate_graph(edited_data_temp[1], rotation, [0.5,0.5]),
                                edited_data_temp[2])
                
                edited_data = [torch.tensor(element) for element in edited_data]
                
                _,_,sols = model(edited_data, return_pi= True)
                sols_list.append(sols)
                
                #get costs on original
                costs = model.problem.get_costs(data, sols)
                costs_list.append(costs)
                
                if index == 0 and i == 0:
                    val_costs = torch.cat(costs_list, dim=0)
                else:
                    val_costs = torch.minimum(val_costs, torch.cat(costs_list, dim=0))
                    
                mean_cost = torch.mean(val_costs)
                
            print("\n")
            print(mean_cost)
       
            
    if model_was_training: model.train()  # restore original model training state
    
    set_decode_type(model, "sampling")
    
    return sols_list, mean_cost

def dilate_and_solve(data, model_path, scales = [0.5], dilation_center = [0,0], rotations=[0]):
        
    embedding_dim = 128
    GRAPH_SIZE = 100
    normalize_cost = False
    save_extras = False
    batch_size = 1000

    model = load_pt_model(model_path,
                         embedding_dim=embedding_dim,
                         graph_size=GRAPH_SIZE,
                         attention_type=0,
                         attention_neighborhood=0,
                         batch_norm=False,
                         size_context=False,
                         normalize_cost=normalize_cost,
                         save_extras=save_extras,
                         device='cuda:0')

    set_decode_type(model, "greedy")
 
    data = [torch.tensor(element) for element in data]
    costs_batches = FastTensorDataLoader(data[0], data[1], data[2], batch_size=batch_size, shuffle=False)
    
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for i, scale in enumerate(scales):
            
            if i == 0:
                edited_data = data
                edited_data_temp = data
            else:
                edited_data = (dilate_graph(data[0], scale, dilation_center),
                                dilate_graph(data[1], scale, dilation_center),
                                data[2])
                edited_data_temp = edited_data
                
                
            for index, rotation in enumerate(rotations):
                
                sols_list = []
                costs_list = []
                
                edited_data = (rotate_graph(edited_data_temp[0], rotation, [0.5,0.5]),
                                rotate_graph(edited_data_temp[1], rotation, [0.5,0.5]),
                                edited_data_temp[2])
                
                edited_data = [torch.tensor(element) for element in edited_data]
                train_batches = FastTensorDataLoader(edited_data[0], edited_data[1], edited_data[2], batch_size=batch_size, shuffle=False)
                
                for batch, cost_batch in tqdm(zip(train_batches,costs_batches), disable=False, desc="Rollout greedy execution"):
                    _,_,sols = model(batch, return_pi= True)
                    sols_list.append(sols)
                    #get costs on original
                    costs = model.problem.get_costs(cost_batch, sols)
                    costs_list.append(costs)
                
                if index == 0 and i == 0:
                    val_costs = torch.cat(costs_list, dim=0)
                else:
                    val_costs = torch.minimum(val_costs, torch.cat(costs_list, dim=0))
                    
                mean_cost = torch.mean(val_costs)
                
            print("\n")
            print(mean_cost)
       
            
    if model_was_training: model.train()  # restore original model training state
    
    set_decode_type(model, "sampling")
    
    return sols_list, mean_cost

# dilated = dilate_graph(data[1], 0.5, [0,0])
# rotated = rotate_graph(data[1], 45, [0.5,0.5])




################################ TEST #########################################
#data path and load
GRAPH_SIZE = 100
# path = 'data/CVRP/vrp'+ str(GRAPH_SIZE)+'_test_seed1234.pkl'
# data = read_from_old_pickle(path)
VAL_SET_PATH = 'C:/Users/inbox/Desktop/Validation_dataset_100.pkl'
data = read_from_pickle(VAL_SET_PATH)

model_name = "mid/3/model/model_checkpoint_epoch_97_dense_entmax_noShuffle_originalEncoder_50_50_2022-01-02"
model_path = 'C:/Users/inbox/Desktop/Results/' + model_name
    

#dilation = 1.2
rotation = 270
rotations = [0, 90, 180, 270]
dilations = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 2]


# data = (rotate_graph(data[0], rotation, [0.5,0.5]),
#                 rotate_graph(data[1], rotation, [0.5,0.5]),
#                 data[2])

sols, cost = dilate_and_solve(data, model_path, dilations, [0.5,0.5], rotations = rotations)

data_path= "C:/Users/inbox/Desktop/200/1/"

#sols, cost = dilate_and_solve_instances(data_path, model_path, dilations, [0.5,0.5], rotations = rotations)




