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
GRAPH_SIZE = 200
val_batch_size = 128

model_name = "mid/baseline_checkpoint_epoch_65_mix50-100_entmax_originalEncoder_50_2022-01-15"

MODEL_PATH = 'C:/Users/inbox/Desktop/Results/' + model_name
VAL_SET_PATH = 'data/CVRP/vrp'+ str(GRAPH_SIZE)+'_test_seed1234.pkl'
#VAL_SET_PATH = 'C:/Users/inbox/Desktop/Validation_dataset_200.pkl'

#model
GRAPH_SIZE = 100

#path_2 = "valsets/Validation_dataset_VRP_20_2021-11-28.pkl"
# Create and save validation dataset
#validation_dataset = read_from_pickle(VAL_SET_PATH)
validation_dataset = read_from_old_pickle(VAL_SET_PATH)

validation_dataset = tuple([torch.tensor(x) for x in validation_dataset])

print(get_cur_time(), 'validation dataset loaded')

save_extras_path = None
#save_extras_path = "model_extras/" + model_name + "/outputs.pkl"
normalize_cost = False
save_extras = False

# Initialize model

model = load_pt_model(MODEL_PATH,
                         embedding_dim=embedding_dim,
                         graph_size=GRAPH_SIZE,
                         attention_type=1,
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

