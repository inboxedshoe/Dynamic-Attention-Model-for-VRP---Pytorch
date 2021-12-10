import torch
from time import gmtime, strftime

from attention_dynamic_model import set_decode_type
from reinforce_baseline import RolloutBaseline
from train import train_model

from utils import get_cur_time
from reinforce_baseline import load_pt_model
from utils import read_from_pickle, read_from_old_pickle
from reinforce_baseline import validate

embedding_dim = 128
GRAPH_SIZE = 50
val_batch_size = 1000

MODEL_PATH = 'checkpts/model_checkpoint_epoch_44_dense04_20_VRP_full_20_2021-12-01'
VAL_SET_PATH = 'data/CVRP/vrp50_test_seed1234.pkl'

#path_2 = "valsets/Validation_dataset_VRP_20_2021-11-28.pkl"
# Create and save validation dataset
#d1 = read_from_pickle(path_2)
validation_dataset = read_from_old_pickle(VAL_SET_PATH)

validation_dataset = tuple([torch.tensor(x) for x in validation_dataset])

print(get_cur_time(), 'validation dataset loaded')

# Initialize model
model = load_pt_model(MODEL_PATH,
                         embedding_dim=embedding_dim,
                         graph_size=GRAPH_SIZE,
                         attention_type=0,
                         attention_neighborhood=0,
                         batch_norm=False,
                         size_context=False,
                         device='cuda:1')

set_decode_type(model, "sampling")
print(get_cur_time(), 'model loaded')

val_cost = validate(validation_dataset, model, val_batch_size)

#print("Validation cost: "+ str(val_cost))

