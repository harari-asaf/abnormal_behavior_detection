import pandas as pd
import os
from signature_matrix import data_to_signatures
from preprocessing import compute_min_max_matrixs, scale_matrix
from metrics_and_scores import  compute_val_test_RMSE, compute_test_anomaly_score, PR_AUC
from trainer import Trainer
from collaborative_anomaly_detector import create_data_for_collaborative_anomaly_detection, compute_dtw_between_drones, agg_DTW_by_drone


def main(config):
 # import data - to recieve the data please contect us via email
 data_folder_path = exp_config['data_dir'] + exp_config['type_of_data'] + '/' + exp_config['scenario'] + '/'
 data_files_ls = os.listdir(data_folder_path)
 test_data, train_data, val_data = [pd.read_csv(data_folder_path + file) for file in data_files_ls]

 # create signature matrices
 train_X, train_labels, train_keys = data_to_signatures(drones_df=train_data, win_size_ls=exp_config['win_size_ls'],
                                                        no_sensors_cols=exp_config['no_sensors_cols'])
 val_X, val_labels, val_keys = data_to_signatures(drones_df=val_data, win_size_ls=exp_config['win_size_ls'],
                                                  no_sensors_cols=exp_config['no_sensors_cols'])
 test_X, test_labels, test_keys = data_to_signatures(drones_df=test_data, win_size_ls=exp_config['win_size_ls'],
                                                     no_sensors_cols=exp_config['no_sensors_cols'])

 # scale - compute min-max on trainig data
 mat_min_X, mat_max_x = compute_min_max_matrixs(train_X)
 # scale datasets
 train_X_scaled = [scale_matrix(iter_data, mat_min_X, mat_max_x) for iter_data in train_X]
 val_X_scaled = [scale_matrix(iter_data, mat_min_X, mat_max_x) for iter_data in val_X]
 test_X_scaled = [scale_matrix(iter_data, mat_min_X, mat_max_x) for iter_data in test_X]

 """self anomaly detector"""
 # train MSCRED
 trainer = Trainer(train_X_scaled, val_X_scaled, test_X_scaled, exp_config)
 model_ephocs_loss_df = trainer.train_MSCRED(random_samplig=True)

 # recuntract
 recuntructed_val_X, recuntructed_test_X = trainer.recunstract_val_and_test()

 # compute anomaly scores and RMSEs for each time step and drone
 val_MSCRED_RMSEs = compute_val_test_RMSE(val_X_scaled, recuntructed_val_X, exp_config)
 test_MSCRED_RMSEs = compute_val_test_RMSE(test_X_scaled, recuntructed_test_X, exp_config)
 test_MSCREDs_anomaly_score = compute_test_anomaly_score(val_X_scaled, recuntructed_val_X, test_X_scaled,
                                                         recuntructed_test_X, test_labels, exp_config)

 """collaborative anomaly detector"""
 # construct new data for the collaborative anomaly detector which contain the results of the self anomaly detector
 val_data_for_collab_anomaly_detector = create_data_for_collaborative_anomaly_detection(val_keys, val_labels,
                                                                                        val_MSCRED_RMSEs, exp_config)
 test_data_for_collab_anomaly_detector = create_data_for_collaborative_anomaly_detection(test_keys, test_labels,
                                                                                         test_MSCRED_RMSEs, exp_config)
 # compute dtw between each pair of drones
 val_dtw_between_drones = compute_dtw_between_drones(val_data_for_collab_anomaly_detector, dtw_win_size=120)
 test_dtw_between_drones = compute_dtw_between_drones(test_data_for_collab_anomaly_detector, dtw_win_size=120)

 # aggregate the dtw using average
 val_keys_with_dtw_and_RMSE = agg_DTW_by_drone(val_dtw_between_drones, val_data_for_collab_anomaly_detector,
                                               drones_in_analysis=5)
 test_keys_with_dtw_and_RMSE = agg_DTW_by_drone(test_dtw_between_drones, test_data_for_collab_anomaly_detector,
                                                drones_in_analysis=5)

 # compute hybrid anomaly score
 test_keys_with_dtw_and_RMSE['DTW_RMSE'] = test_keys_with_dtw_and_RMSE['DTW_dist'] * test_keys_with_dtw_and_RMSE['RMSE']

 # compute PR-AUC for each method
 DTW_PR_AUC = PR_AUC(test_keys_with_dtw_and_RMSE['label'].to_numpy(),
                     test_keys_with_dtw_and_RMSE['DTW_dist'].to_numpy())
 RMSE_PR_AUC = PR_AUC(test_keys_with_dtw_and_RMSE['label'].to_numpy(), test_keys_with_dtw_and_RMSE['RMSE'].to_numpy())
 MSCREDs_anomaly_score_PR_AUC = PR_AUC(test_keys_with_dtw_and_RMSE['label'].to_numpy(), test_MSCREDs_anomaly_score)
 DTW_RMSE_PR_AUC = PR_AUC(test_keys_with_dtw_and_RMSE['label'].to_numpy(),
                          test_keys_with_dtw_and_RMSE['DTW_RMSE'].to_numpy())

 print('============RESULTS for {} =================='.format(exp_config['scenario']))
 print('MSCREDs_anomaly_score_PR_AUC: {} RMSE_PR_AUC: {} DTW_PR_AUC: {} DTW_RMSE_PR_AUC {} '.format(
  MSCREDs_anomaly_score_PR_AUC, RMSE_PR_AUC, DTW_PR_AUC, DTW_RMSE_PR_AUC))


if __name__ == "__main__":
  exp_config = {
   'scenario': 'followpath',  # ,#'followpath',#'simple',#'avoidance',#'survey',#
   'data_dir': 'data/',
   'type_of_data': 'CROSS',  # 'CROSS',  # NAIVE,
   'model_results_output_folder': 'model/', # dir for saving model wights
   'no_sensors_cols': ['Unnamed: 0', 'drone', 'update_step', 'iter', 'script_time', 'sim_time', 'anomaly_type', # columns that do not contains sensory data
                       'scenario', 'original_iter', 'latitude', 'longitude', 'altitude', 'collision'],

   'win_size_ls': [120, 60, 30],
    'sample_factor': 2,
   # MSCRED parameters
   'step_max': 5,
   'sensor_n': 18,
   'sensor_m': 18,
   'scale_n': 3,
   'learning_rate': 0.00005,
   'beta': 1,
   'scale_down_filter': 1,
   'ephoces_num': 40,
   # if to concatnate other drone to X
   'drop_rate': 0.0  # dropuot
  }

  main(exp_config)

