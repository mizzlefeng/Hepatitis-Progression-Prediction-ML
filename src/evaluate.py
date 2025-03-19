# -*- coding:utf-8 -*-
# @FileName  :evaluate.py
# @Time      :2024/11/20 16:37:45
# @Author    :mizzle

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    f1_score,
    auc,
    precision_recall_curve,
    accuracy_score,
    roc_curve,
    cohen_kappa_score,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


def g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm[1, 1] / cm[1, :].sum()
    tnr = cm[0, 0] / cm[0, :].sum()
    gmean = np.sqrt(tpr * tnr)
    return gmean


def evaluate_model(y_true, y_pred, y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    ppv = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    gmean = g_mean(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    FP = cm[0][1]

    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    evaluate_dic = {
        "acc": acc,
        "ppv": ppv,
        "sensitivity": sensitivity,
        "f1": f1,
        "roc_auc": roc_auc,
        "gmean": gmean,
        "kappa": kappa,
        "npv": npv,
        "specificity": specificity,
        "fpr": fpr,
        "tpr": tpr,
    }

    return evaluate_dic


def calculate_aupr(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    return auc(recall, precision)


def calculate_npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    return npv


def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return specificity


def bootstrap_ci(
    y_true,
    y_pred,
    y_pred_prob,
    metric_func,
    use_proba=False,
    n_bootstrap=1000,
    alpha=0.05,
):

    np.random.seed(42)
    n_samples = len(y_true)
    metric_values = []
    if use_proba:
        original_metric = metric_func(y_true, y_pred_prob)
    else:
        original_metric = metric_func(y_true, y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)
    # Bootstrap
    for _ in tqdm(range(n_bootstrap)):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_pred_prob_boot = y_pred_prob[indices]
        if use_proba:
            metric_boot = metric_func(y_true_boot, y_pred_prob_boot)
        else:
            metric_boot = metric_func(y_true_boot, y_pred_boot)
        metric_values.append(metric_boot)

    z0 = np.sum(np.array(metric_values) < original_metric) / n_bootstrap
    z_alpha = norm.ppf(1 - alpha / 2)

    lower_percentile = 100 * norm.cdf(2 * z0 - z_alpha)
    upper_percentile = 100 * norm.cdf(2 * z0 + z_alpha)

    ci_lower = np.percentile(metric_values, lower_percentile)
    ci_upper = np.percentile(metric_values, upper_percentile)

    return original_metric, ci_lower, ci_upper, metric_values


def evaluate_model_CI(y_true, y_pred, y_pred_prob):
    metrics = {
        "acc": (accuracy_score, False),
        "ppv": (lambda y, y_pred: precision_score(y, y_pred, zero_division=0), False),
        "sensitivity": (recall_score, False),
        "specificity": (calculate_specificity, False),
        "f1": (f1_score, False),
        "roc_auc": (lambda y, y_pred_prob: roc_auc_score(y, y_pred_prob), True),
        "gmean": (g_mean, False),
        "kappa": (cohen_kappa_score, False),
        "npv": (calculate_npv, False),
    }

    results = {}
    for name, (func, use_proba) in metrics.items():
        original_metric, ci_lower, ci_upper, metric_values = bootstrap_ci(
            y_true, y_pred, y_pred_prob, func, use_proba=use_proba
        )
        results[name] = {
            "value": original_metric,
            "95% CI": (ci_lower, ci_upper),
            "all value": metric_values,
        }

    return results
