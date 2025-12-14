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
from sklearn.metrics import f1_score, precision_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
import torch
import torch.optim as optim
import torch.nn as nn
from src.common import setup_seed, get_best_para_from_optuna
from model.deep_model import FCN, CNN, train_evaluate

warnings.filterwarnings("ignore")

storage_name = "postgresql://postgres:123...@127.0.0.1/hepatitis"
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
setup_seed(42)
MODE = '6M'

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

result_path = "../result/03experiment/"

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

def g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm[1, 1] / cm[1, :].sum()
    tnr = cm[0, 0] / cm[0, :].sum()
    gmean = np.sqrt(tpr * tnr)
    return gmean

def evaluate_model_vote(y_true, y_pred):
    ppv = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    gmean = g_mean(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    
    evaluate_dic = {
        "ppv": ppv, "f1": f1, "gmean": gmean, "npv": npv
    }
    return evaluate_dic

def train_evaluate_models_vote(model_list, X, y):
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
    results_per_threshold = {}

    for i in range(5):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+i)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_pred_dic = {}
            for model_name, model in models.items():
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
                    
                    _, y_pred, y_pred_prob = train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test, epochs)
                    y_pred_dic[model_name] = y_pred.flatten()
                
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
                    y_pred_dic[model_name] = y_pred.flatten()
                
                else:
                    ml_model = model
                    ml_model.fit(X_train, y_train)
                    y_pred = ml_model.predict(X_test)
                    y_pred_dic[model_name] = y_pred

            for threshold in range(1, len(model_list) + 1):
                votes = np.sum(list(y_pred_dic.values()), axis=0) >= threshold
                metrics = evaluate_model_vote(y_test, votes)
                if threshold not in results_per_threshold:
                    results_per_threshold[threshold] = {metric: [] for metric in metrics}
                for metric, value in metrics.items():
                    results_per_threshold[threshold][metric].append(value)

    all_metrics = []
    for threshold, metrics in results_per_threshold.items():
        metrics_summary = {f"{metric}_mean": np.mean(values) for metric, values in metrics.items()}
        metrics_summary.update({f"{metric}_std": np.std(values) for metric, values in metrics.items()})
        metrics_summary['Voted Threshold'] = f"Voted_{threshold}"
        all_metrics.append(metrics_summary)

    metrics_df = pd.DataFrame(all_metrics)
    
    cols = ['Voted Threshold', 'f1_mean', 'f1_std', 'gmean_mean', 'gmean_std', 'npv_mean', 'npv_std', 'ppv_mean', 'ppv_std']
    
    metrics_df = metrics_df[cols]
    metrics_df.to_excel(os.path.join(result_path, f"01voted_combinations_{MODE}.xlsx"), index=False)

    return metrics_df

metrics_df = train_evaluate_models_vote(model_list, X, y)
single_model = pd.read_excel(f"../result/01experiment/01_1feature_group_roc/CIF+STCF_{MODE}.xlsx")
cols = ['Model', 'f1_mean', 'f1_std', 'gmean_mean', 'gmean_std', 'npv_mean', 'npv_std', 'ppv_mean', 'ppv_std']
    
metrics_df.rename(columns={"Voted Threshold":"Model"},inplace=True)
vote_model = metrics_df[cols]
single_model = single_model[cols]
merge_df = pd.concat([vote_model, single_model], axis=0)
results = pd.DataFrame()
results['Model'] = merge_df['Model']
for col in merge_df.columns:
    if '_mean' in col:
        metric = col.split('_mean')[0]
        mean_col = col
        std_col = metric + '_std'
        results[metric] = merge_df.apply(lambda row: f"{row[mean_col]:.3f}±{row[std_col]:.3f}", axis=1)
results = results[['Model', 'f1', 'gmean', 'npv', 'ppv']]
results.to_csv(os.path.join(result_path, f"01voted_merge_{MODE}.csv"), index=False, encoding="utf-8-sig")



