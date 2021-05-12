# import
import pandas as pd
import numpy as np
import os
from signature_matrix import data_to_signatures
from preprocessing import compute_min_max_matrixs, scale_matrix, adjust_y_or_labels_to_fit_model_output
from metrics_and_scores import  compute_val_test_RMSE, compute_test_anomaly_score
from trainer import Trainer
config = {
'scenario': 'followpath',#'followpath',#'simple',#'avoidance',#'survey',#
'data_dir':'data/', #'data/',# super dir of the simulated data
 'type_of_data': 'CROSS',  # NAIVE
#"""folders to save data and results:"""
'final_data_output_folder': 'seqs_data/',
'model_results_output_folder':'model/',
'no_sensors_cols': ['Unnamed: 0','drone', 'update_step', 'iter', 'script_time', 'sim_time','anomaly_type','scenario','original_iter','latitude', 'longitude', 'altitude','collision','original_iter'],
'hyper_parameters_exp_name': 'win_size_120',

 'win_size_ls':  [120, 60, 30],
 'sample_factor': 2,# 2 # take every "sample_factor" record from raw dtat
 'num_of_drones': 5,
 'numOfcomparisonDronesforTrainig':1,
 'numOfcomparisonDronesforTesting':1,
 'num_of_comp_drones': 1, # number of compromised drones
 'save': True, # save signature matrix  before trianing?
 'load': False,
# model parameters
  'step_max':5,
  'sensor_n':18,
  'sensor_m':18,
  'scale_n':3,
  'learning_rate':0.00005,
 'beta':1,
  'scale_down_filter': 1,
  'ephoces_num': 1,
     # if to concatnate other drone to X
    'drop_rate': 0.0 # dropuot
}


# import data
data_folder_path = config['data_dir']+config['type_of_data']+'/'+config['scenario']+'/'
data_files_ls = os.listdir(data_folder_path)
test_data, train_data, val_data = [pd.read_csv(data_folder_path+file) for file in data_files_ls ]

# create signature matrices

train_X,train_labels, train_keys  =  data_to_signatures(drones_df=train_data[:500],win_size_ls=config['win_size_ls'],no_sensors_cols=config['no_sensors_cols'])
val_X, val_labels , val_keys =  data_to_signatures(drones_df=val_data[:500],win_size_ls=config['win_size_ls'],no_sensors_cols=config['no_sensors_cols'])
test_X, test_labels , test_keys =  data_to_signatures(drones_df=test_data[:10000],win_size_ls=config['win_size_ls'],no_sensors_cols=config['no_sensors_cols'])
# scale
# compute min-max on trainig data
mat_min_X, mat_max_x = compute_min_max_matrixs(train_X)
# scale datasets
train_X_scaled = [scale_matrix(iter_data, mat_min_X, mat_max_x) for iter_data in train_X]
val_X_scaled = [scale_matrix(iter_data, mat_min_X, mat_max_x) for iter_data in val_X]
test_X_scaled = [scale_matrix(iter_data, mat_min_X, mat_max_x) for iter_data in test_X]


"""self anomaly detector"""
"""TRAIN"""
trainer = Trainer(train_X_scaled, val_X_scaled, test_X_scaled,config)
model_ephocs_loss_df = trainer.train_MSCRED(random_samplig=True)
# predict
recuntructed_val_X, recuntructed_test_X = trainer.recunstract_val_and_test()

# compute anomaly scores and RMSEs
val_MSCRED_RMSEs =  compute_val_test_RMSE(val_X_scaled,recuntructed_val_X,config)
test_MSCRED_RMSEs = compute_val_test_RMSE(test_X_scaled,recuntructed_test_X,config)
test_MSCREDs_anomaly_score = compute_test_anomaly_score(val_X_scaled,recuntructed_val_X,test_X_scaled,recuntructed_test_X,test_labels,config)

#
# """collaborative anomaly detector"""
# create_keys_df_with_dtw(MSCRED_val_Y_for_comparison,recunstract_val_ls, val_keys,val_label, step_max,sample_factor,win_size)
#
#
# cola_val = create_data_for_collaborative_anomaly_detection(val_X, val_keys, val_labels,val_MSCRED_RMSEs)
# cola_test = create_data_for_collaborative_anomaly_detection(test_X, test_keys, test_labels,test_MSCRED_RMSEs)

# compute PR-AUC and create data for the collaborative anomaly detector





# use

#