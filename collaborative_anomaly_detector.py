from sklearn.preprocessing import StandardScaler
from pathos.multiprocessing import ProcessingPool
import time
import multiprocessing
import pandas as pd
import numpy as np
from tslearn.metrics import dtw_path
from dtw import *
from preprocessing import adjust_keys_df_for_comparison, adjust_y_or_labels_to_fit_model_output
from signature_matrix import data_to_drones_dfs

def compute_DTW_to_each_drone(drones_df_ls,win_size,no_sensors_cols, per_series=False,process_gps = True,use_scaler=True):
    print('Start compute DTW')

    dataset = pd.concat(drones_df_ls)
    dataset = dataset.sort_values(['iter', 'update_step', 'drone']).reset_index(drop=True)
    drones = dataset.drone.unique()
    numOfDrones = len(drones)

    start = time.time()
    # iter = '0simple'
    # dataset_iteri = dataset.loc[dataset['iter'] == iter, :]
    iters = dataset.iter.unique()
    # create empty df for results

    # itearte over iterartions
    def compute_DTW_on_iter(dataset,iter, numOfDrones, drones, per_series=True):
        print('iter: ',iter)
        dtw_results_dict = {'iter': [], 'update_step': [], 'drone': [], 'comparison_drone': [], 'DTW_dist': []}
        # print('iter: ',iter )
        dataset_iter = dataset.loc[dataset['iter'] == iter, :]
        # cut the df by current update step-win size
        update_step_ls = dataset_iter.update_step.unique()
        # num of features (all columns - no sensor columns and label
        num_of_features = dataset_iter.shape[1] - len(no_sensors_cols + ['label'])
        # iterate over time steps
        for update_step in update_step_ls:
            current_seq = dataset_iter.loc[(dataset_iter['update_step'] <= update_step) &
                                           (dataset_iter['update_step'] > (update_step - win_size))]
            # iterte over drones
            for droneIidx in range(numOfDrones):
                currentDrone = drones[droneIidx]
                currentDroneDf = current_seq.loc[current_seq.drone == currentDrone, :]
                # drop irrelevant cols and convert to numpy
                currentDroneNp = currentDroneDf.drop(no_sensors_cols + ['label'], 1).to_numpy()
                if use_scaler: scaled_currentDroneNp =  StandardScaler().fit_transform(currentDroneNp)
                else: scaled_currentDroneNp = currentDroneNp
                for droneJidx in range(numOfDrones):
                    # dont compare drone to itself
                    if (droneIidx >= droneJidx): continue
                    # print(droneIidx, droneJidx)
                    otherDrone = drones[droneJidx]
                    otherDroneDf = current_seq.loc[current_seq.drone == otherDrone, :]
                    otherDroneNp = otherDroneDf.drop(no_sensors_cols + ['label'], 1).to_numpy()
                    if use_scaler: scaled_otherDroneNp = StandardScaler().fit_transform(otherDroneNp)
                    else: scaled_otherDroneNp = otherDroneNp

                    """compute DTW"""

                    if per_series:  # compute between each pair of series, return list

                        dist = [dtw_path(scaled_currentDroneNp[:, i], scaled_otherDroneNp[:, i])[1] for i in
                                range(num_of_features)]
                        dist = np.array(dist)
                    else:
                        # path, dist = dtw_path(scaled_currentDroneNp, scaled_otherDroneNp)
                        path = ''
                        dist = dtw(scaled_currentDroneNp,scaled_otherDroneNp,window_type="sakoechiba",window_args ={'window_size':60}).distance
                    # print('Iter {} updatestep {} DroneI {} DroneJ {} DTW {}'.format(iter,update_step,currentDrone, otherDrone, dist))
                    # save results of current drone
                    dtw_results_dict['iter'].append(iter);
                    dtw_results_dict['update_step'].append(update_step);
                    dtw_results_dict['drone'].append(currentDrone)
                    dtw_results_dict['comparison_drone'].append(otherDrone);
                    dtw_results_dict['DTW_dist'].append(dist)  # ; dtw_results_dict['DTW_path'].append(path)
                    # save results of other drone
                    dtw_results_dict['iter'].append(iter);
                    dtw_results_dict['update_step'].append(update_step);
                    dtw_results_dict['drone'].append(otherDrone)
                    dtw_results_dict['comparison_drone'].append(currentDrone);
                    dtw_results_dict['DTW_dist'].append(dist)  # ; dtw_results_dict['DTW_path'].append(path)


        print('iter done: ', iter)
        return dtw_results_dict
    workers = multiprocessing.cpu_count()
    print('Number of workers: ', workers)
    workers = np.min([workers,len(iters)])
    pool = ProcessingPool(workers)
    list_of_iters_dict = list(pool.map(lambda iter: compute_DTW_on_iter(dataset,iter,numOfDrones,drones,per_series), iters))
    pool.close(); pool.join(); pool.terminate(); pool.clear()
    # from list of dicts to one dict
    dtw_results_dict = {'iter': [], 'update_step': [], 'drone': [], 'comparison_drone': [], 'DTW_dist': []}
    [dtw_results_dict[result_key].append(value) for dict in list_of_iters_dict for result_key, list in dict.items() for value in list]


    print('time took: ', time.time() - start)

    dtw_results_df = pd.DataFrame.from_dict(dtw_results_dict)
    dtw_results_df = dtw_results_df.sort_values(['iter', 'update_step', 'drone']).reset_index(drop=True)

    dtw_results_df_after_removal_ls = []

    return dtw_results_df

def rolling_mean_on_col(df, win_size, col_name):
    """recisve df anc col, returns the df with new col named: col_name_roll"""
    num_of_drones = len(df.drone.unique())
    new_col_name = col_name + '_roll'
    df[new_col_name] = df[col_name]
    for drone in range(1, num_of_drones + 1):

        for iter in df.iter.unique():
            iter_cond = df.iter == iter
            drone_cond = df.drone == drone
            df.loc[iter_cond & drone_cond, new_col_name] = df.loc[
                iter_cond & drone_cond, new_col_name].rolling(win_size).mean().fillna(0)

    return df


def create_data_for_collaborative_anomaly_detection(keys, labels, RMSE_np, config):
    MSCRED_keys_for_comparison = adjust_keys_df_for_comparison(keys, config['step_max'], config['sample_factor'])
    data_with_labels_and_RMSE = MSCRED_keys_for_comparison.copy()
    MSCRED_label_for_comparison = adjust_y_or_labels_to_fit_model_output(labels, config['step_max'], return_as_np=True)

    data_with_labels_and_RMSE['label'] = MSCRED_label_for_comparison
    data_with_labels_and_RMSE['RMSE'] = RMSE_np
    return data_with_labels_and_RMSE

def compute_dtw_between_drones(data_with_labels_and_RMSE,dtw_win_size=120):

    data_with_labels_and_RMSE  = rolling_mean_on_col(data_with_labels_and_RMSE, win_size=20, col_name='RMSE')

    drones_df_ls = data_to_drones_dfs(data_with_labels_and_RMSE)

    no_sensors_cols = ['drone', 'update_step', 'iter']
    dtw_between_drones = compute_DTW_to_each_drone(drones_df_ls, dtw_win_size, per_series=False, process_gps=False,
                                                   no_sensors_cols=no_sensors_cols,use_scaler=True)

    return dtw_between_drones

def agg_DTW_by_drone(dtw_results_df,data_for_collab_anomaly_detector,keep_KNN_DTW_dist=False,KNN_for_DTW_dist =2,drones_in_analysis=5):
    if drones_in_analysis < 5:
        # drones_in_analysis = 4
        ls_of_drones = list(range(0, 4))
        drone_iter_step_ls = dtw_results_df[['iter','update_step','drone']].drop_duplicates().apply(lambda x:''.join([str(i) for i in list(x)]),1).to_list()
        dtw_results_df['iter_step_drone'] =  dtw_results_df[['iter','update_step','drone']].apply(lambda x:''.join([str(i) for i in list(x)]),1)
        # sp;it into list of dfs, each df with 4 records, record for each comparison drone
        list_of_dfs = [dtw_results_df.loc[dtw_results_df.iter_step_drone==drone_iter_step,:].reset_index(drop=True) for drone_iter_step in drone_iter_step_ls ]
        np.random.seed(0)
        indexs_to_keep = np.random.choice(ls_of_drones, drones_in_analysis - 1, replace=False)

        list_of_dfs = [df.iloc[indexs_to_keep,:] for df in list_of_dfs]
        dtw_results_df = pd.concat(list_of_dfs)

    if keep_KNN_DTW_dist:
        dtw_results_df = dtw_results_df.groupby(['iter', 'update_step', 'drone'])['DTW_dist'].nsmallest(KNN_for_DTW_dist).reset_index()

    anomaly_by_keys = dtw_results_df.groupby(['iter', 'update_step', 'drone'], as_index=False).mean()
    keys_with_dtw_and_RMSE = anomaly_by_keys.sort_values(['iter', 'drone', 'update_step']).reset_index(drop=True)

    keys_with_dtw_and_RMSE['label'] = data_for_collab_anomaly_detector.label
    keys_with_dtw_and_RMSE['RMSE'] = data_for_collab_anomaly_detector.RMSE
    return keys_with_dtw_and_RMSE
