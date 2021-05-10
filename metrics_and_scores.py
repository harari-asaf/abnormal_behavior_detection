import numpy as np
import pandas as pd
from sklearn.metrics import auc,precision_recall_curve, f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error

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


def compute_RMSE(Y, pre):
    iters_num = len(Y)
    MAE_ls = [np.sqrt(((Y[iter] - pre[iter]) ** 2).mean(axis=(1,2,3))) for iter in range(iters_num)]
    MAE_np = np.concatenate(MAE_ls, axis=(0))
    return MAE_np

def metrics(y, pre, score,precison_recall_auc=False,only_auc=False):
    f = f1_score(y, pre)

    recall = recall_score(y, pre)

    precision = precision_score(y, pre)


    if precison_recall_auc:
        precision_ls, recall_ls, thresholds = precision_recall_curve(y, score)
        print('auc thresholds:' ,thresholds[:4])

        "comnpute f for each thr"
        F_ls =((precision_ls * recall_ls) / (precision_ls + recall_ls))*2
        thr_idx = np.argmax(F_ls)
        best_thr = thresholds[thr_idx]
        print('best threshold by auc analysis: ',best_thr )
        # # remove thr = 0
        # precision_ls = precision_ls[1:]; precision_ls = precision_ls[1:]
        p_r_auc = auc(recall_ls , precision_ls)
        AUC = p_r_auc
    else:
        AUC = roc_auc_score(y, score)
    res = {'f': f, 'precision': precision, 'recall': recall, 'auc': AUC,
           'accuracy': accuracy_score(y, pre)}
    if only_auc:
        res = {'auc': AUC}
    print(res)
    return res

def compute_thr_and_metrics_on_drones(anomaly_score, labels,thr):

    # scoreVSanomaly.loc[scoreVSanomaly['label'] == 0, 'score'].quantile(0.9) #
    # stats.percentileofscore(scoreVSanomaly.loc[scoreVSanomaly['label'] == 0, 'score'], thr)
    if thr == 0: pre = (anomaly_score > thr) * 1
    else: pre = (anomaly_score >= thr) * 1
    """prediction accuracy on all drones and time steps"""
    print('accuracy on positive iters for all drones SEPARATELY in each time steps ')
    res = metrics(labels, pre,anomaly_score,precison_recall_auc=True)

    return res, pre