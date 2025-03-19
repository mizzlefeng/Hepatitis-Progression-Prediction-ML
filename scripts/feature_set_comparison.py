import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..src.common import setup_seed,get_best_para_from_optuna
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.metrics import auc
import warnings
import itertools
import optuna
import torch
import torch.optim as optim
import torch.nn as nn
import os
import seaborn as sns
from model.deep_model import FCN, CNN,train_evaluate
from ..src.evaluate import evaluate_model
warnings.filterwarnings("ignore")

CV_RESULT = ""
LOGFILE = ""

storage_name = "postgresql://postgres:xxx@127.0.0.1/hep_f1"
result_path = ""
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
setup_seed(42)

def my_svm(**params):
    return SVC(probability=True, **params)

def my_lgb(**params):
    return LGBMClassifier(verbose=-1, **params)

model_dict = {
    "LR": LogisticRegression,
    "SVM": my_svm,
    "NB": GaussianNB,
    "KNN": KNeighborsClassifier,
    "RF": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": my_lgb,
    "FCN": FCN,
    "CNN": CNN
}

def train_and_evaluate_model(model, X, y, fold, color):
    setup_seed(42)
    model_name = model[0]
    skf = StratifiedKFold(n_splits=fold, shuffle=True)
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []
    
    metrics_all = {
        "acc": [], "ppv": [], "sensitivity": [], "f1": [],
        "roc_auc": [], "mcc": [], "aupr": [], "gmean": [],
        "npv": [], "kappa": [], "specificity": [],
    }
    auc_list = []
    for i in range(10):
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if model_name == 'FCN':
                best_params = model[1]
                lr = best_params["lr"]
                n_layers = best_params["n_layers"]
                hidden_layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]
                activation_func = best_params["activation_func"]
                optimizer_name = best_params["optimizer"]

                net = FCN(input_dim=X.shape[1], output_dim=1, hidden_layers=hidden_layers, activation_func=activation_func).to(device)
                optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()

                _,y_pred,y_pred_prob=train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test)

            elif model_name == 'CNN':
                best_params = model[1]

                lr = best_params["lr"]
                n_layers = best_params["n_layers"]
                n_filters = [best_params[f"n_filters_l{i}"] for i in range(n_layers)]
                kernel_size = best_params["kernel_size"]
                activation_func = best_params["activation_func"]
                optimizer_name = best_params["optimizer"]

                net = CNN(n_features=X.shape[1], output_dim=1, n_layers=n_layers, n_filters=n_filters, kernel_size=kernel_size, activation_func=activation_func).to(device)
                optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()

                _,y_pred,y_pred_prob = train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test)

            else:
                ml_model = model[1]
                ml_model.fit(X_train, y_train)
                y_pred = ml_model.predict(X_test)
                y_pred_prob = ml_model.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_pred_prob)
            for key in metrics_all.keys():
                metrics_all[key].append(metrics[key])
            
            fpr, tpr = metrics["fpr"], metrics["tpr"]
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr_all.append(interp_tpr)

            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)
  
    metrics_summary = {f"{metric}_mean": np.mean(values) for metric, values in metrics_all.items()}
    metrics_summary.update({f"{metric}_std": np.std(values) for metric, values in metrics_all.items()})

    mean_tpr = np.mean(interp_tpr_all, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics_summary['roc_auc_mean']
    if model_name == 'FCN':
        model_name = 'FCNN'
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2, label=f"{model_name} (AUC = {mean_auc:.4f})")

    return metrics_summary

def save_results_to_excel(results, filename):
    df = pd.DataFrame(results)

    cols = ['Model','f1_mean','f1_std','roc_auc_mean','roc_auc_std','npv_mean','npv_std',
            'gmean_mean','gmean_std','kappa_mean','kappa_std',
            'acc_mean','acc_std','ppv_mean','ppv_std','sensitivity_mean','sensitivity_std','specificity_mean','specificity_std']
    df = df[cols]
    df.to_excel(filename, index=False)

def compare_models(models, X, y, fold, plot_name, file_path):

    colors = sns.color_palette("tab10", len(models))
    sns.set_style("ticks")
    results = []
    plt.figure(figsize=(8, 6))

    for model, color in zip(models.items(), colors):
        metrics_summary = train_and_evaluate_model(model, X, y, fold, color)
        metrics_summary["Model"] = model[0]
        results.append(metrics_summary)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plot_name = plot_name.replace(" ", "")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold',c='k')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold',c='k')
    plt.tick_params(axis='x', colors='k')
    plt.tick_params(axis='y', colors='k')
    plt.tick_params(axis='x', labelcolor='k')
    plt.tick_params(axis='y', labelcolor='k')
    plt.title(f'{plot_name}', fontsize=16, fontweight='bold',c='k')
    plt.savefig(os.path.join(result_path,f"{plot_name}_ROC.tif"),dpi=300,bbox_inches="tight")
    plt.savefig(os.path.join(result_path,f"{plot_name}_ROC.pdf"),dpi=600,bbox_inches="tight")
    # plt.show()

    save_results_to_excel(results,os.path.join(file_path,f"{plot_name}.xlsx"))

plt.rcParams['font.family'] = 'Arial'
model_list = ["SVM", "LR", "NB", "KNN", "XGBoost", "RF", "LightGBM", "FCN", "CNN"]

data = pd.read_csv('')
y = data['Label']
clinical_features = ['Gender', 'ALT', 'AST', 'Globulin', 'DBIL', 'IBIL', 'AFP', 'DNA load', 'HBsAg', 'HBeAg_COI']
specific_features = ['SFU', 'HBsAg1_T', 'HBsAg2_T', 'HBpol1_T', 'HBpol2_T', 'HBx1_T', 'HBx2_T', 'HBeAg1_T', 'HBeAg2_T']
treat_features = ['ThSched', 'ADV', 'ETV', 'PEG_IFN', 'TAF', 'TDF', 'TFV', 'TMF', 'UnusedD']
feature_dict = {
    "CIF": clinical_features,  # Clinical index features CIF
    "STCF": specific_features,  # Specific T cell features  STCF
    "TPF": treat_features  # Treatment program features TPF
}

plot_feature_group_dic = {
    "clinical_features": "CIF",
    "specific_features": "STCF",
    "treat_features": "TPF",
}

feature_names = list(feature_dict.keys())
combinations_1 = list(itertools.combinations(feature_names, 1))
combinations_2 = list(itertools.combinations(feature_names, 2))
combinations_3 = list(itertools.combinations(feature_names, 3))
all_combinations = combinations_1 + combinations_2 + combinations_3

for combo in all_combinations:
    combined_features = []
    for group in combo:
        combined_features.extend(feature_dict[group])
    feature_group = ' + '.join([g for g in combo])
    X = data[combined_features]
    models = {}
    for model_name in model_list:
        study_name = feature_group+"_"+model_name
        best_params = get_best_para_from_optuna(study_name)

        if model_name in ["FCN", "CNN"]:
            models[model_name] = best_params
        else:
            model_constructor = model_dict[model_name]
            model = model_constructor(**best_params)
            models[model_name] = model
    
    compare_models(models, X, y, fold=5, plot_name=feature_group, file_path=result_path)