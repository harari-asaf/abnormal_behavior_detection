from sklearn.preprocessing import StandardScaler
import gc
from pathos.multiprocessing import ProcessingPool
import multiprocessing
import pandas as pd
import numpy as np



def data_to_drones_dfs(data):
    """recives: data df with n drones
       return: list of n dfs, df for each drone"""
    drones_arr =  data['drone'].unique()
    drones_df_ls = [data.loc[data['drone'] == drone , :] for drone in  drones_arr]
    # cheack if all have same observations
    df_shapes = [df.shape for df in drones_df_ls]
    shape = df_shapes[0]
    print('all drones have same observations:' ,all([shape == df_shape for df_shape in df_shapes]))
    return drones_df_ls


def seq_to_sig_matrix(seq):
    """Receives numpy mat seqs with SHAPE (win_size, num_of_features)
     Returns numpy signature matrix with SHAPE (num_of_features, num_of_features) """
    sig_mat = np.dot(seq.T,seq) / seq.shape[0]
    return sig_mat

def scale_position_within_each_iter(drones_df_ls):
    position_axis = ['position_x', 'position_y', 'position_z']
    for drone_df in drones_df_ls:
        iter_ls = drone_df.iter.unique()
        for iter in iter_ls:

                axises_data = drone_df.loc[drone_df.iter == iter,position_axis].to_numpy()
                scaled_axises_data = axises_data-axises_data[0]
                # scaled_rational_axises_data = scaled_axises_data-scaled_axises_data[0]
                drone_df.loc[drone_df.iter == iter,position_axis] = scaled_axises_data


    return drones_df_ls


def create_sigmats_3_scales(dataset,no_sensors_cols,win_size_ls,normalize_each_seq=False,warm_up_time_points=''):
    """recives df of the data,
     no_sensors_cols (ls): the columns that doesnt represent sensors
     win_size_ls (ls): win sizes to produce (each one will be a channel in reverse order)
     warm_up_time_points

    returns list of  representations (sigmat) with n  dim (number of channels) for each scale, for each iter
    - X(PADED TO THE MAX LENGTH) shape = (num of seqs, length of seq, num of sensors/features)
    - y and
    - keys ('drone', 'update_step', 'iter') for later identification """


    # compute y - if one of the recorsed is anomaly, all the sequnce classified as anomaly
    iter_ls = dataset.iter.unique()

    def create_sigmats_of_one_iter(dataset, iteri):
        # get current iter
        dataset_iteri = dataset.loc[dataset['iter'] == iteri, :]
        # get list of update steps
        update_step_ls = dataset_iteri.update_step.to_list()
        step_sig_mat_ls = []

        for update_step in update_step_ls:
            # print('iter: ',iteri,'step: ', update_step)

            win_sig_mat_ls = []
            for win_size in win_size_ls:
                # cut the df by current update step-win size
                current_seq = dataset_iteri.loc[(dataset_iteri['update_step'] <= update_step) &
                                                (dataset_iteri['update_step'] > (update_step - win_size))]
                # drop irrelevant cols and convert to numpy
                current_seq = current_seq.drop(no_sensors_cols + ['label'], 1).to_numpy()
                if normalize_each_seq:
                    current_seq = StandardScaler().fit_transform(current_seq)
                # convert to sig mat
                current_seq_sig_mat = seq_to_sig_matrix(current_seq)
                # add to thr ls -each elemnt with different win size
                win_sig_mat_ls.append(current_seq_sig_mat)

            # stack the 3 win size (scale) togather as channels
            # stacked_mats_different_scale = np.stack(win_sig_mat_ls)
            # add to step ls
            step_sig_mat_ls.append(win_sig_mat_ls)
        # stack all steps
        # stacked_mats_of_iter = np.stack(step_sig_mat_ls)
        # add to iter ls
        iter_sig_mat_np = np.array(step_sig_mat_ls)
        iter_sig_mat_np = np.rollaxis(np.array(iter_sig_mat_np), 1, 4)
        return {'sig_mat':iter_sig_mat_np, 'keys':dataset_iteri[['drone', 'update_step', 'iter']], 'labels':dataset_iteri.label.to_numpy()}


    workers = multiprocessing.cpu_count()
    print('Number of workers: ', workers)
    pool = ProcessingPool(workers)
    list_of_iters_dict = pool.map(lambda iter: create_sigmats_of_one_iter(dataset, iter), iter_ls)
    pool.close() ; pool.join() ; pool.terminate() ; pool.clear()

    iters_sig_mat_ls = [iter_dict['sig_mat'] for iter_dict in list_of_iters_dict]
    iters_lables_ls = [iter_dict['labels'] for iter_dict in list_of_iters_dict]
    iters_keys_ls = [iter_dict['keys'] for iter_dict in list_of_iters_dict]

    print('shape of first iter X {} shape of first iter labels {} shape keys {}'.format(iters_sig_mat_ls[0].shape, iters_lables_ls[0].shape, iters_keys_ls[0].shape))

    return iters_sig_mat_ls, iters_lables_ls, iters_keys_ls

def create_moving_seqs_matrix(data_set,no_sensors_cols,win_size,warm_up_time_points=''):
    """recives df of the data,
    returns only sensory data
    - X(PADED TO THE MAX LENGTH) shape = (num of seqs, length of seq, num of sensors/features)
    - y and
    - keys ('drone', 'update_step', 'iter') for later identification """


    drones_ls = data_set.drone.to_list()
    iter_ls = data_set.iter.to_list()
    index_ls = data_set.index.to_list()
    update_step_ls = data_set.update_step.to_list()
    # each time point represented  by the former points in the same iter and drone
    data_set_df_ls = [data_set.loc[(data_set['iter'] == iteri) &
                               # (data_set['drone'] == drone) &
                               (data_set['update_step'] <= update_step) &
                                   (data_set['update_step'] > (update_step-win_size))
                      , :]
                  for iteri,drone,update_step   in zip(iter_ls,drones_ls,update_step_ls)]

    # save the key of the chosen sequences ('drone', 'update_step', 'iter')
    keys = data_set[['drone', 'update_step', 'iter']]
    if warm_up_time_points != '':
        data_set_df_ls = data_set_df_ls[warm_up_time_points:]
        keys = keys[warm_up_time_points:]

    # keep only X columns and convert to numpy each df
    seqs_np_ls = [df.drop(no_sensors_cols+['label'], 1).to_numpy() for df in data_set_df_ls]
    # pad small seqs to fit window size
    pad_seqs_np_ls = [np.pad(seqs_np,[(win_size-len(seqs_np), 0),(0, 0)])
                      if len(seqs_np) <win_size
                      else seqs_np
                      for seqs_np in seqs_np_ls ]


    # convert each seq to signature matrix and create obe matrix from all signature matrix's
    X_data_set = np.stack([seq_to_sig_matrix(x) for x in pad_seqs_np_ls])
    # compute y - if one of the recorsed is anomaly, all the sequnce classified as anomaly
    y_data_set = data_set.label.to_numpy()

    print('shape X {} shape y {} shape keys {}'.format(X_data_set.shape, y_data_set.shape, keys.shape))

    return X_data_set, y_data_set, keys

def combine_drones_data_each_neighbor_is_channel(drones_seqs_mat_ls):
    """Recives list of n drones data, each data is a list of iterations data, each iteration data composed of (sigmats, labels, keys)   """
    num_of_drones = len(drones_seqs_mat_ls)
    current_drone_sig_ls, other_drones_data_ls, keys_ls, labels_ls = [], [], [], [], []

    # input 1 is the ith drone mat, input 2 is the mean of others
    for i in range(num_of_drones):
        i_drones_seqs_mat = drones_seqs_mat_ls[i][0]
        # other drones matrixes
        other_drones_seqs_mat = [drones_seqs_mat_ls[j][0] for j in range(num_of_drones) if i != j]
        other_drones_seqs_mat = np.rollaxis(np.array(other_drones_seqs_mat), 0, 4)

        current_drone_sig_ls.append(i_drones_seqs_mat)
        # save other drones matrixs (each matrix is a channel)
        other_drones_data_ls.append(other_drones_seqs_mat)
        labels_ls.append(drones_seqs_mat_ls[i][1])  # save the label of the drone
        # save same keys for input 1 and two
        keys_ls.append(drones_seqs_mat_ls[i][2])

    current_drone_sig = np.concatenate(current_drone_sig_ls, axis=0)
    # save other drones matrixs
    other_drones_data = np.concatenate(other_drones_data_ls, axis=0)
    # anomaly label 0/1
    labels = np.concatenate(labels_ls, axis=0)
    # keys for later analysis - DF
    keys_df = pd.concat(keys_ls)

    return current_drone_sig, other_drones_data, labels, keys_df

def combine_drones_data_for_MSCRED(drones_seqs_mat_ls,numOfcomparisonDrones=1):
    """Recives list of n drones data, each data is a list of iterations data, each iteration data composed of [sigmats, labels, keys]
    Returns:
        current_drone_sigmat (np): (iters*drones,steps (variable),sigmatI,sigmatJ,scale)
         other_drone_sigmat (np) : (iters*drones,steps (variable),sigmatI,sigmatJ,scale) of random drone from its neighbors
         labels (np): (iters*steps*drones,1) True/False
          keys (pd): (Drone, Iter, step)
          """
    num_of_drones = len(drones_seqs_mat_ls)
    num_of_iters = len(drones_seqs_mat_ls[0][0])

    current_drone_sigmat_ls, other_drone_sigmat_ls, keys_ls, labels_ls = [], [], [], []

    for iterIdx in range(num_of_iters):
        for droneIdx in range(num_of_drones):

            current_drone_keys = drones_seqs_mat_ls[droneIdx][2][iterIdx].copy()
            # labels
            current_drone_labels = drones_seqs_mat_ls[droneIdx][1][iterIdx].copy()
            # Sigmat
            current_drone_sigmat = drones_seqs_mat_ls[droneIdx][0][iterIdx].copy()
            other_drones = list(set(range(num_of_drones)) - set([droneIdx]))

            for neighborDroneIdx in other_drones[:numOfcomparisonDrones]:


                # save neighbor drone number in current keys
                neighbor_keys =  drones_seqs_mat_ls[neighborDroneIdx][2][iterIdx].copy()
                current_neighbor_drone_keys = current_drone_keys.copy()
                current_neighbor_drone_keys['neighbor_drone'] = neighbor_keys.drone.values
                neighbor_drone_sigmat = drones_seqs_mat_ls[neighborDroneIdx][0][iterIdx].copy()


                # append to list
                current_drone_sigmat_ls.append(current_drone_sigmat)
                other_drone_sigmat_ls.append(neighbor_drone_sigmat)
                labels_ls.append(current_drone_labels)  # save the label of the drone
                keys_ls.append(current_neighbor_drone_keys)

    combined_current_drone_sigmats, combined_other_drone_sigmat = np.array(current_drone_sigmat_ls), np.array(other_drone_sigmat_ls)
    labels, keys = np.array(labels_ls), pd.concat(keys_ls)
    # # add neighbor drone id to the iter key
    # keys['iter'] = keys.iter +keys.drone.astype(str) + keys.neighbor_drone.astype(str)

    return combined_current_drone_sigmats, combined_other_drone_sigmat, labels, keys

def data_to_signatures(drones_df,no_sensors_cols,win_size_ls=[] ,normalize_each_seq = False):
    """receives swarm data as df converts each df to sequences numpy matrix with rolling win size

    returns input and output for Autoencoder and anomaly detection algorithm X  , y , labels (for anomaly detection) and keys for later analysis"""
    print('create signatures - it may take a while')
    drones_df_ls = data_to_drones_dfs(drones_df)
    # scale positions columns to start from 0
    drones_df_ls = scale_position_within_each_iter(drones_df_ls)
    # create seqs matrix, labels and keys for each drone in swarm with win size. return list(drones) of tuples (list(X), list(labels), list(keys)) of list iters data shape (drones, 3,iters)
    print('Create sigmat, keys and labels to each drone')
    # create seqs matrix, labels and keys for each drone in swarm with win size. return list of tuples (X, y, keys)
    drones_seqs_mat_ls = [create_sigmats_3_scales (drone_df,no_sensors_cols,win_size_ls,normalize_each_seq) for drone_df in drones_df_ls]
    print('Combine data in iterations for MSCRED model')
    # create data for mscred by iter, list of lists each elemnt is a drone-iter data
    drone_sig, _, labels, keys_df = combine_drones_data_for_MSCRED(drones_seqs_mat_ls)
    return drone_sig,labels,  keys_df




