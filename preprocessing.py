import numpy as np


def compute_min_max_matrixs(data_set):
 mat_min_ls = [iter_data.min(axis=(0)) for iter_data in data_set]
 mat_min_X = np.array(mat_min_ls).min(axis=(0))
 mat_max_ls = [iter_data.max(axis=(0)) for iter_data in data_set]
 mat_max_x = np.array(mat_max_ls).max(axis=(0))
 return mat_min_X, mat_max_x


def scale_matrix(mat, mat_min='', mat_max='',only_min_max_mats = False):
    if mat_max == '':
        print('compute min max')
        mat_min = mat.min(axis=(0), keepdims=True)
        mat_max = mat.max(axis=(0), keepdims=True)
        if only_min_max_mats: return mat_min, mat_max

        scaled_mat = mat - mat_min
        min_max_diff = mat_max - mat_min
        scaled_mat = np.divide(scaled_mat, min_max_diff)
        return scaled_mat, mat_min, mat_max
    scaled_mat = mat - mat_min
    min_max_diff = mat_max - mat_min
    scaled_mat = np.divide(scaled_mat, min_max_diff)
    return scaled_mat


def adjust_y_or_labels_to_fit_model_output(set_to_adjust, step_max, return_as_np=True):
    """
    recives y (np) or lables (np) and remove from each iter the first step_max steps as in model output
    """
    set_after_removing_first_steps_ls = [iter_data[step_max:] for iter_data in set_to_adjust]
    if return_as_np:
        set_rebuilt_as_numpy_array = np.concatenate(set_after_removing_first_steps_ls, axis=(0))
        return set_rebuilt_as_numpy_array
    else:
        return set_after_removing_first_steps_ls

def adjust_keys_df_for_comparison(keys, step_max,sample_factor):
    remove_first_recs = ~keys.update_step.isin(list(range(0, step_max * sample_factor, sample_factor)))
    MSCRED_keys_for_comparison = keys.loc[remove_first_recs, :].reset_index(drop=True)
    MSCRED_keys_for_comparison = MSCRED_keys_for_comparison.drop('neighbor_drone',1)
    return MSCRED_keys_for_comparison



test_X_for_comparison = adjust_y_or_labels_to_fit_model_output(test_X, config['step_max'], return_as_np=False)
test_labels_for_comparison = adjust_y_or_labels_to_fit_model_output(test_labels, config['step_max'])
test_keys_for_comparison = adjust_keys_df_for_comparison(test_keys,config['step_max'],config['sample_factor'])
