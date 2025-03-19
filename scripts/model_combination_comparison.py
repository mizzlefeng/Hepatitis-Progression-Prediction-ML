import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..src.common import init_logger,setup_seed
from ..src.common import get_best_para_from_optuna
from ..src.evaluate import evaluate_model
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import warnings
import itertools
import torch.optim as optim
import torch.nn as nn
import os
import torch
import seaborn as sns
from model.deep_model import FCN, CNN,train_evaluate
warnings.filterwarnings("ignore")

storage_name = "postgresql://postgres:xxx@127.0.0.1/hep_f1"
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

result_path = "../result/02experiment/"

model_list = ["LR","SVM", "NB", "KNN", "RF", "XGBoost", "LightGBM", "FCN", "CNN"]
metric_map = {
    "acc": "Acc", "ppv": "PPV", "sensitivity": "Sensitivity", "f1": "F1",
    "roc_auc": "AUC", "gmean": "G-mean", "kappa": "Kappa", "npv": "NPV", 
    "specificity": "Specificity", "fpr": "FPR", "tpr": "TPR"
}

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

combined_features = feature_dict['CIF']+feature_dict['STCF']
X = data[combined_features]

feature_group = "CIF + STCF"

def train_evaluate_models(model_list, X, y):
    models = {}

    for model_name in model_list:
        study_name = feature_group+"_"+model_name
        best_params = get_best_para_from_optuna(study_name,storage_name)
        if model_name in ["FCN", "CNN"]:
            models[model_name] = best_params
        else:
            model_constructor = model_dict[model_name]
            model = model_constructor(**best_params)
            models[model_name] = model
    
    setup_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    metrics_all = {
        "acc": [], "ppv": [], "sensitivity": [], "f1": [],
        "roc_auc": [], "mcc": [], "aupr": [], "gmean": [],
        "npv": [], "kappa": [], "specificity": [],
    }
    for i in range(10):
        # print(i)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_pred_prob_dic = {}
            for model_name,model in models.items():
                if model_name == 'FCN':
                    best_params = model
                    lr = best_params["lr"]
                    n_layers = best_params["n_layers"]
                    hidden_layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]
                    activation_func = best_params["activation_func"]
                    optimizer_name = best_params["optimizer"]

                    net = FCN(input_dim=X.shape[1], output_dim=1, hidden_layers=hidden_layers, activation_func=activation_func).to(device)
                    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
                    criterion = nn.BCEWithLogitsLoss()

                    _,y_pred,y_pred_prob=train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test)
                    y_pred_prob_dic[model_name] = y_pred_prob.flatten()
                
                elif model_name == 'CNN':
                    best_params = model
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
                    y_pred_prob_dic[model_name] = y_pred_prob.flatten()
                
                else:
                    ml_model = model
                    ml_model.fit(X_train, y_train)
                    y_pred = ml_model.predict(X_test)
                    y_pred_prob = ml_model.predict_proba(X_test)[:, 1]
                    y_pred_prob_dic[model_name] = y_pred_prob

            y_pred_prob_list = list(y_pred_prob_dic.values())
            y_pred_prob = np.mean(y_pred_prob_list, axis=0)
            y_pred = (y_pred_prob > 0.5).astype(int)
            metrics = evaluate_model(y_test, y_pred, y_pred_prob)
            for key in metrics_all.keys():
                metrics_all[key].append(metrics[key])

    metrics_summary = {f"{metric}_mean": np.mean(values) for metric, values in metrics_all.items()}
    metrics_summary.update({f"{metric}_std": np.std(values) for metric, values in metrics_all.items()})

    return metrics_summary
    
def save_results_to_excel(results, filename):
    df = pd.DataFrame(results)

    cols = ['Model','f1_mean','f1_std','roc_auc_mean','roc_auc_std','npv_mean','npv_std',
            'aupr_mean','aupr_std','gmean_mean','gmean_std','kappa_mean','kappa_std','mcc_mean','mcc_std',
            'acc_mean','acc_std','ppv_mean','ppv_std','sensitivity_mean','sensitivity_std','specificity_mean','specificity_std']
    df = df[cols]
    df.to_excel(filename, index=False)

plt.rcParams['font.family'] = 'Arial'
combinations_2 = list(itertools.combinations(model_list, 2))
combinations_3 = list(itertools.combinations(model_list, 3))
all_combinations = combinations_2 + combinations_3
all_combinations

len(all_combinations)

results = []
n = 0
LOGFILE = f""
logger,file_handler = init_logger(LOGFILE)
for combo in all_combinations:
    n+=1
    model_list = list(combo)
    metrics_summary = train_evaluate_models(model_list, X, y)
    metrics_summary["Model"] = '+'.join([g for g in combo])
    results.append(metrics_summary)

save_results_to_excel(results,os.path.join(result_path,f"01consensus_combinations.xlsx"))

file_handler.close()

