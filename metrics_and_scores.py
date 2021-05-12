import numpy as np
import pandas as pd
from sklearn.metrics import auc,precision_recall_curve, f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
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
    print('auc thresholds:' ,thresholds[:4])

    "comnpute f for each thr"
    F_ls =((precision_ls * recall_ls) / (precision_ls + recall_ls))*2
    thr_idx = np.argmax(F_ls)
    best_thr = thresholds[thr_idx]

    # # remove thr = 0
    # precision_ls = precision_ls[1:]; precision_ls = precision_ls[1:]
    p_r_auc = auc(recall_ls , precision_ls)
    AUC = p_r_auc

    print(AUC)
    return {'auc': AUC}
#
# def compute_thr_and_metrics_on_drones(anomaly_score, labels,thr):
#
#     # scoreVSanomaly.loc[scoreVSanomaly['label'] == 0, 'score'].quantile(0.9) #
#     # stats.percentileofscore(scoreVSanomaly.loc[scoreVSanomaly['label'] == 0, 'score'], thr)
#     if thr == 0: pre = (anomaly_score > thr) * 1
#     else: pre = (anomaly_score >= thr) * 1
#     """prediction accuracy on all drones and time steps"""
#     print('accuracy on positive iters for all drones SEPARATELY in each time steps ')
#     res = metrics(labels, pre,anomaly_score,precison_recall_auc=True)
#
#     return res, pre



def compute_test_anomaly_score(val_X,recuntructed_val_X,test_X,recuntructed_test_X,labels,config):
    val_X_for_comparison = adjust_y_or_labels_to_fit_model_output(val_X, config['step_max'], return_as_np=False)
    test_X_for_comparison = adjust_y_or_labels_to_fit_model_output(test_X, config['step_max'], return_as_np=False)
    labels_for_comparison = adjust_y_or_labels_to_fit_model_output(labels, config['step_max'], return_as_np=True)

    MASCREDs_anomaly_score = compute_anomaly_score_per_pixel_on_ls(Y_val=val_X_for_comparison,
                                                                  pre_val=recuntructed_val_X,
                                                                  Y_test=test_X_for_comparison,
                                                                  pre_test=recuntructed_test_X, beta=config['beta'])
    print('PR-AUC FOR SELF-ANOMALY-DETECTOR:')
    AUC = PR_AUC(labels_for_comparison,MASCREDs_anomaly_score)

    return MASCREDs_anomaly_score

#
# def compute_anomaly_score_and_rmse(val_X,recuntructed_val_X,test_X,recuntructed_test_X,config):
#     val_X_for_comparison = adjust_y_or_labels_to_fit_model_output(val_X, config['step_max'], return_as_np=False)
#     test_X_for_comparison = adjust_y_or_labels_to_fit_model_output(test_X, config['step_max'], return_as_np=False)
#
#
#     MASCRED_anomaly_score = compute_anomaly_score_per_pixel_on_ls(Y_val=val_X_for_comparison,
#                                                                   pre_val=recuntructed_val_X,
#                                                                   Y_test=test_X_for_comparison,
#                                                                   pre_test=recuntructed_test_X, beta=config['beta'])
#
#     MSCRED_rmse = compute_RMSE(test_X_for_comparison, recuntructed_test_X)
#
#     return MASCRED_anomaly_score, MSCRED_rmse
#
#
#
# labels_for_comparison = adjust_y_or_labels_to_fit_model_output(labels, config['step_max'])
# keys_for_comparison = adjust_keys_df_for_comparison(keys,config['step_max'],config['sample_factor'])