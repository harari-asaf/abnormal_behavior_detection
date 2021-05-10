config = {
'scenario': '1',#'followpath',#'simple',#'avoidance',#'survey',#
'scenarios_data_dir':'fixed_data_Jan21/', #'data/',# super dir of the simulated data
#"""folders to save data and results:"""
'final_data_output_folder': 'seqs_data/',
'model_results_output_folder':'models/MSCRED/',
# """sub folders:"""
'name_of_model': '3_3_baseline_fixed_data_LOO_RESIDUAL_ANALYSIS_all_360',##'3_3_baseline_fixed_data_LOO_RESIDUAL_ANALYSIS_all',
'model_exp_name': '3_3_baseline_fixed_data_LOO_RESIDUAL_ANALYSIS_all_360',#'3_3_baseline_fixed_data_LOO_RESIDUAL_ANALYSIS_all' ,# 'CA_AutoEncoder_one_drone_by_iter', # for results and wights folder
'no_sensors_cols': ['drone', 'update_step', 'iter', 'script_time', 'sim_time','anomaly_type','scenario','original_iter','latitude', 'longitude', 'altitude'],

'hyper_parameters_exp_name': 'win_size_120',
'normalize_each_seq': False,
'use_dtw_as_other_drone_data': False,
'use_other_drone_as4dim':False,
 'win_size_ls':  [120, 60, 30],
 'win_size': 90,
 'sample_factor': 2,# 2 # take every "sample_factor" record from raw dtat
 'num_of_drones': 5,
 'use_other_drone': False,
 'byDTW_comparison_drone':False,
 'numOfcomparisonDronesforTrainig':1, # how many comparison drones for the training (when no other drones, use 1)
 'numOfcomparisonDronesforTesting':1, # how many comparison drones for the testing (when no other drones, use 1)
 'num_of_comp_drones': 1,
 'save': False,
 'load': True,

 'use_test_set':True,
 'LOO':True,
 'tune_thr_with_test':False,
 'train_ratio': 0.6,
 'val_from_train_ratio': 0.2,
# model parameters
  'step_max':5,
  'sensor_n':18,
  'sensor_m':18*2,
  'scale_n':3,
  'learning_rate':0.00005,
  'scale_down_filter': 1,
  'ephoces_num': 40,
     # if to concatnate other drone to X
    'drop_rate': 0.0
}