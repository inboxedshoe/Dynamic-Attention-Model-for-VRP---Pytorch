
import torch
from attention_dynamic_model import AttentionDynamicModel, set_decode_type
from reinforce_baseline import RolloutBaseline
from train import train_model

from time import strftime, gmtime
from utils import create_data_on_disk, get_cur_time



# Params of model
SAMPLES = 128*10000 #512# 128*10000
BATCH = 128
START_EPOCH = 0
END_EPOCH = 100
FROM_CHECKPOINT = False
embedding_dim = 128
LEARNING_RATE = 0.0001
ROLLOUT_SAMPLES = 10000
NUMBER_OF_WP_EPOCHS = 1
GRAD_NORM_CLIPPING = 1.0
BATCH_VERBOSE = 50
VAL_BATCH_SIZE = 1000
VALIDATE_SET_SIZE = 10000
SEED = 1234
GRAPH_SIZE = 50
FILENAME = 'VRP_{}_{}'.format(GRAPH_SIZE, strftime("%Y-%m-%d", gmtime()))

model_pt = AttentionDynamicModel(embedding_dim).cuda()
set_decode_type(model_pt, "sampling")
print(get_cur_time(), 'model initialized')

validation_dataset = create_data_on_disk(GRAPH_SIZE,
                                         VALIDATE_SET_SIZE,
                                         is_save=True,
                                         filename=FILENAME,
                                         is_return=True,
                                         seed = SEED)
print(get_cur_time(), 'validation dataset created and saved on the disk')


# Initialize optimizer
optimizer = torch.optim.Adam(params=model_pt.parameters(), lr=LEARNING_RATE)

# Initialize baseline
baseline = RolloutBaseline(model_pt,
                           wp_n_epochs = NUMBER_OF_WP_EPOCHS,
                           epoch = 0,
                           num_samples=ROLLOUT_SAMPLES,
                           filename = FILENAME,
                           from_checkpoint = FROM_CHECKPOINT,
                           embedding_dim=embedding_dim,
                           graph_size=GRAPH_SIZE
                           )
print(get_cur_time(), 'baseline initialized')

torch.cuda.empty_cache()


train_model(optimizer,
            model_pt,
            baseline,
            validation_dataset,
            samples = SAMPLES,
            batch = BATCH,
            val_batch_size = VAL_BATCH_SIZE,
            start_epoch = START_EPOCH,
            end_epoch = END_EPOCH,
            from_checkpoint = FROM_CHECKPOINT,
            grad_norm_clipping = GRAD_NORM_CLIPPING,
            batch_verbose = BATCH_VERBOSE,
            graph_size = GRAPH_SIZE,
            filename = FILENAME
            )
