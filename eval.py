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

model_name = "model_checkpoint_epoch_86_embeddingFixed_mixed_50_50_2021-12-25"

MODEL_PATH = 'C:/Users/inbox/Desktop/Results/' + model_name
VAL_SET_PATH = 'data/CVRP/vrp100_test_seed1234.pkl'

#path_2 = "valsets/Validation_dataset_VRP_20_2021-11-28.pkl"
# Create and save validation dataset
#d1 = read_from_pickle(path_2)
validation_dataset = read_from_old_pickle(VAL_SET_PATH)

validation_dataset = tuple([torch.tensor(x) for x in validation_dataset])

print(get_cur_time(), 'validation dataset loaded')

save_extras_path = None
#save_extras_path = "model_extras/" + model_name + "/outputs.pkl"
normalize_cost = True
save_extras=False

# Initialize model

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


val_cost = validate(validation_dataset, model, val_batch_size, save_extras_path)

#print("Validation cost: "+ str(val_cost))

