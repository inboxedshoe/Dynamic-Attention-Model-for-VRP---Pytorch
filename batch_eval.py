# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 22:44:16 2022

@author: inbox
"""

import torch
from attention_dynamic_model import set_decode_type


from utils import get_cur_time
from reinforce_baseline import load_pt_model
from utils import read_from_old_pickle
from reinforce_baseline import validate

import os
import re

from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


val_sizes = [100]
model_folders = [1]

#model

save_extras_path = None
normalize_cost = False
save_extras = False
val_batch_size = 1000
embedding_dim = 128
attention_neighborhood = 0

print(get_cur_time(), 'validation dataset loaded')

for model_folder in model_folders:
        
    #instances path
    instance_path = 'C:/Users/inbox/Desktop/Results/mid/' + str(model_folder) + '/'
    save_type = 'model'
    files = os.listdir(instance_path + save_type + '/')
    files.sort(key=natural_keys)
    
    for val_size in val_sizes:
        
        #val_set
        GRAPH_SIZE = val_size
        VAL_SET_PATH = 'data/CVRP/vrp'+ str(GRAPH_SIZE)+'_test_seed1234.pkl'
        validation_dataset = read_from_old_pickle(VAL_SET_PATH)
        validation_dataset = tuple([torch.tensor(x) for x in validation_dataset])
        
        val_costs = []
        
        for file in tqdm(files):
            
           # model_name = "model_checkpoint_epoch_99_entmax_originalEncoder_50_50_2022-01-02"
            MODEL_PATH = instance_path + save_type + '/' + file
            
            # Initialize model
            GRAPH_SIZE = 100
            
            model = load_pt_model(MODEL_PATH,
                                     embedding_dim=embedding_dim,
                                     graph_size=GRAPH_SIZE,
                                     attention_type=1,
                                     attention_neighborhood = attention_neighborhood,
                                     batch_norm=False,
                                     size_context=False,
                                     normalize_cost=normalize_cost,
                                     save_extras=save_extras,
                                     device='cuda:0')
            
            
            set_decode_type(model, "sampling")
            #print(get_cur_time(), 'model loaded')
            
            
            val_cost = validate(validation_dataset, model, val_batch_size, save_extras_path, disable_tqdm=True, print_res=False)
            val_costs.append(val_cost)
        
        #print("Validation cost: "+ str(val_cost))
         
        val_costs = [round(x.item(), 3) for x in val_costs]
        
        with open(instance_path + save_type +'_costs_'+ str(val_size)+'.pkl', 'wb') as f:
            pickle.dump(val_costs, f)
        
        with open(instance_path + save_type +'_costs_'+ str(val_size)+'.pkl', 'rb') as f:
            b = pickle.load(f)
            
        
        
            
        plt.figure(dpi=200)
        plt.grid()
        plt.plot(b)
        plt.xlabel("epoch")
        plt.ylabel("Validation Cost")
        
        plt.savefig(instance_path + save_type +'_costs_'+ str(val_size))    
        plt.show()    
    
    
    
    
    
    
    
    
