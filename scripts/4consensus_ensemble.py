import os
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from src.common import init_logger, setup_seed, get_best_para_from_optuna
from src.evaluate import evaluate_model
from model.deep_model import FCN, CNN, train_evaluate

warnings.filterwarnings("ignore")

MODE = '6M'
OUTPUTDIR = "../result/03experiment/"
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

model_list = ["LR", "SVM", "NB", "KNN", "RF", "XGBoost", "LightGBM", "FCNN", "CNN", "TabPFN"]
metric_map = {
    "ppv": "PPV", "f1": "F1","roc_auc": "AUC", "gmean": "Gmean","npv": "NPV", "fpr": "FPR", "tpr": "TPR", "aupr": "AUPR"
}

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

combined_features = feature_dict['CIF']+feature_dict['STCF']
X = data[combined_features]

feature_group = "CIF + STCF"

def train_evaluate_models(model_list, X, y):
    models = {}

    for model_name in model_list:
        study_name = feature_group + "_" + model_name + "_" + MODE
        best_params = get_best_para_from_optuna(study_name, storage_name)

        if model_name in ["FCNN", "CNN"]:
            models[model_name] = best_params
        else:
            model_constructor = model_dict[model_name]
            model = model_constructor(**best_params)
            models[model_name] = model

    setup_seed(42)
    metrics_all = {
        "ppv": [], "f1": [],"roc_auc": [], "aupr": [], "gmean": [],"npv": []
    }
    for i in range(5):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+i)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_pred_prob_dic = {}
            for model_name,model in models.items():
                if model_name == 'FCNN':
                    best_params = model
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
                    y_pred_prob_dic[model_name] = y_pred_prob.flatten()
                
                elif model_name == 'CNN':
                    best_params = model
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
                    y_pred_prob_dic[model_name] = y_pred_prob.flatten()
                
                else:
                    ml_model = model
                    ml_model.fit(X_train, y_train)
                    y_pred = ml_model.predict(X_test)
                    y_pred_prob = ml_model.predict_proba(X_test)[:, 1]
                    y_pred_prob_dic[model_name] = y_pred_prob

            y_pred_prob_list = list(y_pred_prob_dic.values())
            y_pred_prob = np.mean(y_pred_prob_list, axis=0)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            metrics = evaluate_model(y_test, y_pred, y_pred_prob)
            for key in metrics_all.keys():
                metrics_all[key].append(metrics[key])

    metrics_summary = {f"{metric}_mean": np.mean(values) for metric, values in metrics_all.items()}
    metrics_summary.update({f"{metric}_std": np.std(values) for metric, values in metrics_all.items()})

    return metrics_summary
    
def save_results_to_excel(results, filename):
    df = pd.DataFrame(results)

    cols = ['Model','f1_mean','f1_std','roc_auc_mean','roc_auc_std','npv_mean','npv_std','aupr_mean','aupr_std','gmean_mean','gmean_std','ppv_mean','ppv_std']
    df = df[cols]
    df.to_excel(filename, index=False)

plt.rcParams['font.family'] = 'Arial'
combinations_2 = list(itertools.combinations(model_list, 2))
combinations_3 = list(itertools.combinations(model_list, 3))
all_combinations = combinations_2 + combinations_3

results = []
n = 0
LOGFILE = os.path.join(OUTPUTDIR,f'01consensus_combinations_{MODE}.log')
if os.path.exists(LOGFILE):
    os.remove(LOGFILE)
logger,file_handler = init_logger(LOGFILE)
for combo in all_combinations:
    n+=1
    model_list = list(combo)
    logger.info(f"Combination {n}: {model_list}...................")
    metrics_summary = train_evaluate_models(model_list, X, y)
    metrics_summary["Model"] = '+'.join([g for g in combo])
    results.append(metrics_summary)

save_results_to_excel(results,os.path.join(OUTPUTDIR,f"01consensus_combinations_{MODE}.xlsx"))

file_handler.close()

