import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, matthews_corrcoef,
    roc_curve
)


def search_best_threshold(y_true, y_prob):
    """
    使用 Youden’s J statistic (TPR - FPR) 搜索最佳阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_th = thresholds[best_idx]
    return best_th


def calculate_metrics(y_true, y_prob):
    """
    根据预测概率自动选择分类阈值，输出 SEN / PRE / F1 / ACC / SPE / MCC
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # ===== 搜索最佳阈值 =====
    best_th = search_best_threshold(y_true, y_prob)

    # ===== 应用最佳阈值 =====
    y_pred = (y_prob >= best_th).astype(int)

    # ===== 指标（全部基于最佳阈值）=====
    sensitivity = recall_score(y_true, y_pred)                     # SEN
    precision = precision_score(y_true, y_pred, zero_division=0)   # PRE
    F1_score = f1_score(y_true, y_pred)                            # F1
    accuracy = accuracy_score(y_true, y_pred)                      # ACC
    specificity = recall_score(y_true, y_pred, pos_label=0)        # SPE
    mcc = matthews_corrcoef(y_true, y_pred)                        # MCC

    return sensitivity, precision, F1_score, accuracy, specificity, mcc


def get_result(loader, model):
    """
    返回: auc, sen, pre, F1, acc, spe, mcc
    """
    pred, target = [], []
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda().float()
            y = y.cuda().float()
            y_hat = model(x)
            pred.extend(list(y_hat.cpu().numpy()))
            target.extend(list(y.cpu().numpy()))

    auc = roc_auc_score(target, pred)

    # ===== 自动阈值后的分类指标 =====
    sen, pre, F1, acc, spe, mcc = calculate_metrics(target, pred)

    return auc, sen, pre, F1, acc, spe, mcc

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_auc_score
#
#
# def calculate_metrics(y_true, y_pred):
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for i in range(len(y_true)):
#         if y_true[i] == 1 and y_pred[i] >= 0.4:
#             TP += 1
#         if y_true[i] == 0 and y_pred[i] < 0.4:
#             TN += 1
#         if y_true[i] == 0 and y_pred[i] >= 0.4:
#             FP += 1
#         if y_true[i] == 1 and y_pred[i] < 0.4:
#             FN += 1
#
#     # 原有指标
#     sensitivity = TP / (TP + FN + 1e-10)
#     precision = TP / (TP + FP + 1e-10)
#     F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
#
#     # 新增指标
#     accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)  # ACC
#     specificity = TN / (TN + FP + 1e-10)  # SPE
#     # 马修斯相关系数 (MCC)
#     numerator = (TP * TN) - (FP * FN)
#     denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)
#     mcc = numerator / denominator
#
#     return sensitivity, precision, F1_score, accuracy, specificity, mcc
#
#
# def get_result(loader, models):
#     pred, target = [], []
#     models.eval()
#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.cuda().float(), y.cuda().float()
#             y_hat = models(x)
#             pred += list(y_hat.cpu().numpy())
#             target += list(y.cpu().numpy())
#     auc = roc_auc_score(target, pred)
#     # 接收并返回新增的三个指标
#     sen, pre, F1, acc, spe, mcc = calculate_metrics(target, pred)
#     return auc, sen, pre, F1, acc, spe, mcc