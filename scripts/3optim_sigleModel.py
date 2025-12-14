import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (f1_score, precision_score, 
                             confusion_matrix, roc_curve, precision_recall_curve, auc)
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from src.common import setup_seed, get_best_para_from_optuna
from model.deep_model import FCN, CNN, train_evaluate

warnings.filterwarnings("ignore")

OUTPUTDIR = "../result/02experiment/"
MODE = "6M"
storage_name = "postgresql://postgres:123...@127.0.0.1/hepatitis"
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
setup_seed(42)

def my_svm(**params):
    return SVC(probability=True, **params)

def my_lgb(**params):
    return LGBMClassifier(verbose=-1, **params)

def my_tabpfn(**params):
    return TabPFNClassifier(device='cuda',n_jobs=5, **params)

model_dict = {
    "LR": LogisticRegression,
    "SVM": my_svm,
    "NB": GaussianNB,
    "KNN": KNeighborsClassifier,
    "RF": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": my_lgb,
    "FCNN": FCN,
    "CNN": CNN,
    "TabPFN": my_tabpfn,
}

def g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm[1, 1] / cm[1, :].sum()
    tnr = cm[0, 0] / cm[0, :].sum()
    gmean = np.sqrt(tpr * tnr)
    return gmean

def evaluate_model(y_true, y_pred, y_pred_prob):
    ppv = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall_arr, precision_arr)
    gmean = g_mean(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    
    evaluate_dic = {
        "ppv": ppv, "f1": f1, "roc_auc": roc_auc, "aupr": aupr, 
        "gmean": gmean, "npv": npv, "fpr": fpr, "tpr": tpr
    }
    return evaluate_dic

def train_and_evaluate_model(model, X, y, fold, color, roc_color=None):
    setup_seed(42)
    model_name = model[0]

    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []
    
    metrics_all = {"ACC": [], "PPV": [], "F1": [], "AUC": [], "Gmean": [], "NPV": [], "AUPR":[]}
    auc_list = []
    y_true_lst = []
    y_prob_lst = []
    
    for i in range(5):
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42+i)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_true_lst.extend(y_test)

            if model_name == 'FCNN':
                best_params = model[1]
                lr = best_params["lr"]
                n_layers = best_params["n_layers"]
                hidden_layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]
                activation_func = best_params["activation_func"]
                optimizer_name = best_params["optimizer"]
                epochs = best_params['epochs']

                net = FCN(input_dim=X.shape[1], output_dim=1, hidden_layers=hidden_layers, activation_func=activation_func).to(device)
                optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()
                _,y_pred,y_pred_prob=train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test, epochs)
                y_prob_lst.extend(y_pred_prob)

            elif model_name == 'CNN':
                best_params = model[1]

                lr = best_params["lr"]
                n_layers = best_params["n_layers"]
                n_filters = [best_params[f"n_filters_l{i}"] for i in range(n_layers)]
                kernel_size = best_params["kernel_size"]
                activation_func = best_params["activation_func"]
                optimizer_name = best_params["optimizer"]
                epochs = best_params['epochs']

                net = CNN(n_features=X.shape[1], output_dim=1, n_layers=n_layers, n_filters=n_filters, kernel_size=kernel_size, activation_func=activation_func).to(device)
                optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()
                _,y_pred,y_pred_prob=train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test, epochs)
                y_prob_lst.extend(y_pred_prob)

            else:
                ml_model = model[1]
                ml_model.fit(X_train, y_train)
                y_pred = ml_model.predict(X_test)
                y_pred_prob = ml_model.predict_proba(X_test)[:, 1]
                y_prob_lst.extend(y_pred_prob)

            metrics = evaluate_model(y_test, y_pred, y_pred_prob)
            for key in metrics_all.keys():
                metrics_all[key].append(metrics[key])
            
            fpr, tpr = metrics["fpr"], metrics["tpr"]
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr_all.append(interp_tpr)
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)

    metrics_all['y_true'] = y_true_lst
    metrics_all['y_prob'] = y_prob_lst
    metrics_summary = {f"{metric}_mean": np.mean(values) for metric, values in metrics_all.items()}
    metrics_summary.update({f"{metric}_std": np.std(values) for metric, values in metrics_all.items()})

    mean_tpr = np.mean(interp_tpr_all, axis=0)
    mean_tpr[-1] = 1.0

    if roc_color:
        plt.plot(mean_fpr, mean_tpr, color=roc_color, lw=2, 
                label=f"{model_name} (AUC = {metrics_summary['AUC_mean']:.3f})")

    return metrics_summary, metrics_all

def annotate_top_values(df, tol=1e-8):

    result_df = df.copy()
    
    for col in result_df.columns:
        if pd.api.types.is_numeric_dtype(result_df[col]) and col != 'Model':
            non_null = result_df[col].dropna()
            sorted_vals = non_null.sort_values(ascending=False)
            top1 = sorted_vals.iloc[0]
            top2 = sorted_vals.iloc[1]
            
            for idx in result_df.index:
                val = result_df.at[idx, col]

                if val == top1:
                    result_df.at[idx, col] = f"a{val:.3f}"
                elif val == top2:
                    result_df.at[idx, col] = f"b{val:.3f}"
                else:
                    result_df.at[idx, col] = f"{val:.3f}"
    
    return result_df

def save_results_to_excel_annotate(results, filename):
    df = pd.DataFrame(results)

    mean_cols = ['Model','F1_mean','AUC_mean','Gmean_mean','AUPR_mean','NPV_mean','ACC_mean','PPV_mean']
    std_cols = ['Model','F1_std','AUC_std','Gmean_std','AUPR_std','NPV_std','ACC_std','PPV_std']
    need_col = ['Model','F1','AUC','Gmean','AUPR','NPV','PPV','ACC']
    df_mean = df[mean_cols]
    df_mean_transposed = annotate_top_values(df_mean)
    df_std = df[std_cols]
    df_combined = pd.DataFrame()
    for col in need_col:
        if col == 'Model':
            df_combined[col] = df[col]
        else:
            df_combined[col] = [f"{m}±{s:.3f}" for m, s in zip(df_mean_transposed[f"{col}_mean"], df_std[f"{col}_std"])]
    
    df_combined.to_excel(filename, index=False)

def save_results_to_excel_raw(results, filename):
    df = pd.DataFrame(results)

    mean_cols = ['Model','F1_mean','AUC_mean','Gmean_mean','AUPR_mean','NPV_mean','ACC_mean','PPV_mean']
    need_col = ['Model','F1','AUC','Gmean','AUPR','NPV','PPV','ACC']
    df_mean = df[mean_cols]
    df_combined = pd.DataFrame()
    for col in need_col:
        if col == 'Model':
            df_combined[col] = df[col]
        else:
            df_combined[col] = [f"{m:.3f}" for m in df_mean[f"{col}_mean"]]
    
    df_combined.to_excel(filename, index=False)

def save_results_to_excel_detail(results, filename):
    df = pd.DataFrame(results)
    df.to_pickle(filename)

def compare_models(models, X, y, fold, plot_name, file_path):
    colors = sns.color_palette("tab10", len(models))
    
    results = []
    detail_list = []
    
    sns.set_style("ticks")
    plt.figure(figsize=(8, 6), facecolor='none')
    
    for i, (model, color) in enumerate(zip(models.items(), colors)):
        metrics_summary, metrics_all = train_and_evaluate_model(model, X, y, fold, color, roc_color=color)
        metrics_summary["Model"] = model[0]
        results.append(metrics_summary)
        metrics_all["Model"] = model[0]
        detail_list.append(metrics_all)

    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold', c='k')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold', c='k')
    plt.tick_params(axis='x', colors='k')
    plt.tick_params(axis='y', colors='k')
    plt.tick_params(axis='x', labelcolor='k')
    plt.tick_params(axis='y', labelcolor='k')
    plt.title(f'{MODE} ROC Curve', fontsize=16, fontweight='bold', c='k')
    plt.legend(loc="lower right", prop={'weight':'bold', 'size':11}, 
                       labelcolor='black', edgecolor='black', framealpha=1, fancybox=True, 
                       frameon=True, title='Models', 
                       title_fontproperties={'weight':'bold', 'size':12})
    
    plt.savefig(os.path.join(OUTPUTDIR, f"ROC_{MODE}.tif"), format='tif', dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUTDIR, f"ROC_{MODE}.pdf"), format='pdf', dpi=300, bbox_inches="tight")
    
    save_results_to_excel_annotate(results, os.path.join(file_path, f"{MODE}_annotate.xlsx"))
    save_results_to_excel_raw(results, os.path.join(file_path, f"{MODE}_raw.xlsx"))
    save_results_to_excel_detail(detail_list, os.path.join(file_path, f"{MODE}_detail.pkl"))

plt.rcParams['font.family'] = 'Arial'
model_list = ["LR", "SVM", "NB", "KNN", "RF", "XGBoost", "LightGBM", "FCNN", "CNN", "TabPFN"]
data = pd.read_csv('../result/00pre-processing/05final_data-minmax.csv')
y = data[f'{MODE}-Label']

clinical_features = ['Gender', 'ALT', 'AST', 'Albumin', 'GGT', 'DBIL', 'IBIL', 'AFP', 'DNA load', 'HBsAg']
specific_features = ['HBV-T', 'HBsAg-T(pH>7)', 'HBsAg-T(pH≤7)', 'HBpol-T(pH>7)', 'HBpol-T(pH≤7)', 'HBx-T(pH>7)', 'HBx-T(pH≤7)', 'HBeAg-T(pH>7)', 'HBeAg-T(pH≤7)']
treatment_features = ['ThSched', 'ADV', 'ETV', 'PEG-IFN', 'TAF', 'TDF', 'TFV', 'TMF']

feature_dict = {
    "CIF": clinical_features,
    "STCF": specific_features,
    "TPF": treatment_features
}

plot_feature_group_dic = {
    "clinical_features": "CIF",
    "specific_features": "STCF",
    "treatment_features": "TPF",
}

combined_features = feature_dict['CIF']+feature_dict['STCF']
X = data[combined_features]

feature_group = "CIF + STCF"

models = {}
for model_name in model_list:
    study_name = feature_group+"_"+model_name+"_"+MODE
    best_params = get_best_para_from_optuna(study_name, storage_name)

    if model_name in ["FCNN", "CNN"]:
        models[model_name] = best_params
    else:
        model_constructor = model_dict[model_name]
        model = model_constructor(**best_params)
        models[model_name] = model

compare_models(models, X, y, fold=5, plot_name=feature_group, file_path=OUTPUTDIR)