import numpy as np
from sklearn.metrics import auc,precision_recall_curve
from preprocessing import adjust_y_or_labels_to_fit_model_output

def compute_anomaly_score_per_pixel_on_ls(Y_val, pre_val, Y_test, pre_test, beta):
    val_iters_num = len(Y_val)
    # find max error of each iter
    val_max_error_ls  = [np.max(np.abs(Y_val[iter] - pre_val[iter]), axis=0) for iter in range(val_iters_num)]

    # make np array from all max errors
    val_max_error_np =  np.array(val_max_error_ls)#np.concatenate(val_max_error_ls, axis=(0))
    # fins global max error and multiply by beta
    val_pairs_error_max =  beta *np.max(val_max_error_np, axis=0)

    positive_iters_num = len(Y_test)
    positive_pairs_error_ls =[np.abs(Y_test[iter] - pre_test[iter]) for iter in range(positive_iters_num)]

    anomaly_score_ls = [np.sum((positive_pairs_error_ls[iter] > val_pairs_error_max), axis=(1, 2,3)) for iter in range(positive_iters_num)]

    anomaly_score = np.concatenate(anomaly_score_ls, axis=(0))
    return anomaly_score


def compute_val_test_RMSE(Y, pre,config):
    Y = adjust_y_or_labels_to_fit_model_output(Y, config['step_max'], return_as_np=False)
    iters_num = len(Y)
    RMSE_ls = [np.sqrt(((Y[iter] - pre[iter]) ** 2).mean(axis=(1,2,3))) for iter in range(iters_num)]
    RMSE_np = np.concatenate(RMSE_ls, axis=(0))

    return RMSE_np

def compute_RMSE(Y, pre):
    iters_num = len(Y)
    MAE_ls = [np.sqrt(((Y[iter] - pre[iter]) ** 2).mean(axis=(1,2,3))) for iter in range(iters_num)]
    MAE_np = np.concatenate(MAE_ls, axis=(0))
    return MAE_np


def PR_AUC(y, score):
    precision_ls, recall_ls, thresholds = precision_recall_curve(y, score)
    p_r_auc = auc(recall_ls , precision_ls)
    AUC = p_r_auc
    print(AUC)
    return  AUC



def compute_test_anomaly_score(val_X,recuntructed_val_X,test_X,recuntructed_test_X,labels,config):
    val_X_for_comparison = adjust_y_or_labels_to_fit_model_output(val_X, config['step_max'], return_as_np=False)
    test_X_for_comparison = adjust_y_or_labels_to_fit_model_output(test_X, config['step_max'], return_as_np=False)
    labels_for_comparison = adjust_y_or_labels_to_fit_model_output(labels, config['step_max'], return_as_np=True)

    MASCREDs_anomaly_score = compute_anomaly_score_per_pixel_on_ls(Y_val=val_X_for_comparison,
                                                                  pre_val=recuntructed_val_X,
                                                                  Y_test=test_X_for_comparison,
                                                                  pre_test=recuntructed_test_X, beta=config['beta'])
    print('PR-AUC FOR SELF-ANOMALY-DETECTOR:')
    PR_AUC(labels_for_comparison,MASCREDs_anomaly_score)

    return MASCREDs_anomaly_score
