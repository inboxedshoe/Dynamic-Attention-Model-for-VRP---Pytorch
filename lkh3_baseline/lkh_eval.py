#%%

import numpy as np
import sys
sys.platform

#%%

from baseline_lkh3 import BaselineLKH3


#%%

# from utils import create_data_on_disk
# val_data = create_data_on_disk(50,
#                                10,
#                                is_save=True,
#                                filename='small_10_50',
#                                is_return=True,
#                                seed=1234)

#%%

cur_dir = '/mnt/c/Users/Human/rl/Project/attention-learn-to-route-master/AM-D-VRP-Development-master/AM-D-VRP-Development-master/adm_best_vrp_tf/LKH_3_VRP_50_2020-06-10'
val_data_path = 'Validation_dataset_VRP_50_2020-06-09.pkl'
executable_path = '/mnt/c/Users/Human/rl/Project/attention-learn-to-route-master/AM-D-VRP-Development-master/AM-D-VRP-Development-master/adm_best_vrp_tf/LKH-3.0.4/LKH'
lkh3_b = BaselineLKH3(cur_dir,
                      val_data_path,
                      executable_path
                      )

lkh3_b.create_lkh_data()

lkh3_costs, lkh3_paths, lkh3_duration = lkh3_b.run_lkh3()

print(np.mean(np.array(lkh3_costs)))
