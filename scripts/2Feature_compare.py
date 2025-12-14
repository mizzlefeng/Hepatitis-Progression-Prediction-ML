import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve, precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.common import setup_seed
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from src.common import setup_seed, get_best_para_from_optuna
from model.deep_model import FCN, CNN, train_evaluate
import os
import warnings

warnings.filterwarnings("ignore")

# Define paths and constants
CV_RESULT = os.path.join("..", "result", "01experiment")
LOGFILE = os.path.join("..", "result", "01experiment", "01model.log")
MODE = "6M"

storage_name = "postgresql://postgres:123...@127.0.0.1/hepatitis"
result_path = os.path.join("..", "result", "01experiment", "01_1feature_group_roc")
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
setup_seed(42)


def my_svm(**params):
    return SVC(probability=True, **params)


def my_lgb(**params):
    return LGBMClassifier(verbose=-1, **params)


def my_tabpfn(**params):
    return TabPFNClassifier(device="cuda", n_jobs=5, **params)


# Model dictionary
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
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0  # NPV
    
    evaluate_dic = {
        "ppv": ppv, "f1": f1, "roc_auc": roc_auc, "aupr": aupr, 
        "gmean": gmean, "npv": npv, "fpr": fpr, "tpr": tpr
    }
    return evaluate_dic


def train_and_evaluate_model(model_tuple, X, y, fold, color):

    setup_seed(42)
    model_name = model_tuple[0]
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr_all = []
    
    metrics_all = {"ppv": [], "f1": [], "roc_auc": [], "gmean": [], "npv": [], "aupr": []}
    auc_list = []
    
    for i in range(5):
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42 + i)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if model_name == "FCNN":
                best_params = model_tuple[1]
                lr = best_params["lr"]
                n_layers = best_params["n_layers"]
                hidden_layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]
                activation_func = best_params["activation_func"]
                optimizer_name = best_params["optimizer"]
                epochs = best_params["epochs"]

                net = FCN(input_dim=X.shape[1], output_dim=1, hidden_layers=hidden_layers, activation_func=activation_func).to(device)
                optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()
                _, y_pred, y_pred_prob = train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test, epochs)

            elif model_name == "CNN":
                best_params = model_tuple[1]
                lr = best_params["lr"]
                n_layers = best_params["n_layers"]
                n_filters = [best_params[f"n_filters_l{i}"] for i in range(n_layers)]
                kernel_size = best_params["kernel_size"]
                activation_func = best_params["activation_func"]
                optimizer_name = best_params["optimizer"]
                epochs = best_params["epochs"]

                net = CNN(n_features=X.shape[1], output_dim=1, n_layers=n_layers, n_filters=n_filters, kernel_size=kernel_size, activation_func=activation_func).to(device)
                optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
                criterion = nn.BCEWithLogitsLoss()
                _, y_pred, y_pred_prob = train_evaluate(net, criterion, optimizer, X_train, y_train, X_test, y_test, epochs)

            else:
                ml_model = model_tuple[1]
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
    mean_auc = metrics_summary["roc_auc_mean"]
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2, label=f"{model_name} (AUC = {mean_auc:.4f})")

    return metrics_summary


def save_results_to_excel(results, filename):
    df = pd.DataFrame(results)

    cols = ["Model", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std", "npv_mean", "npv_std",
            "gmean_mean", "gmean_std", "ppv_mean", "ppv_std", "aupr_mean", "aupr_std"]
    df = df[cols]
    df.to_excel(filename, index=False)


def compare_models(models, X, y, fold, plot_name, file_path):

    colors = sns.color_palette("tab10", len(models))
    sns.set_style("ticks")
    results = []
    plt.figure(figsize=(8, 6))

    for model_tuple, color in zip(models.items(), colors):
        metrics_summary = train_and_evaluate_model(model_tuple, X, y, fold, color)
        metrics_summary["Model"] = model_tuple[0]
        results.append(metrics_summary)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plot_name = plot_name.replace(" ", "")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2)
    plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold", c="k")
    plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold", c="k")
    plt.tick_params(axis="x", colors="k")
    plt.tick_params(axis="y", colors="k")
    plt.tick_params(axis="x", labelcolor="k")
    plt.tick_params(axis="y", labelcolor="k")
    plt.title(f"{plot_name}", fontsize=16, fontweight="bold", c="k")
    
    legend = plt.legend(loc="lower right", prop={"weight": "bold", "size": 12}, labelcolor="black", 
                       edgecolor="black", framealpha=1, frameon=True, title="Models", 
                       title_fontproperties={"weight": "bold", "size": 13})

    save_results_to_excel(results, os.path.join(file_path, f"{plot_name}_{MODE}.xlsx"))



plt.rcParams["font.family"] = "Arial"
clinical_features = ["Gender", "ALT", "AST", "Albumin", "GGT", "DBIL", "IBIL", "AFP", "DNA load", "HBsAg"]
specific_features = ["HBV-T", "HBsAg-T(pH>7)", "HBsAg-T(pH≤7)", "HBpol-T(pH>7)", "HBpol-T(pH≤7)", 
                     "HBx-T(pH>7)", "HBx-T(pH≤7)", "HBeAg-T(pH>7)", "HBeAg-T(pH≤7)"]
treatment_features = ["ThSched", "ADV", "ETV", "PEG-IFN", "TAF", "TDF", "TFV", "TMF"]

feature_dict = {
    "CIF": clinical_features,  # Clinical index features
    "STCF": specific_features,  # Specific T cell features
    "TPF": treatment_features  # Treatment program features
}


# Generate all feature group combinations
feature_names = list(feature_dict.keys())
combinations_1 = list(itertools.combinations(feature_names, 1))
combinations_2 = list(itertools.combinations(feature_names, 2))
combinations_3 = list(itertools.combinations(feature_names, 3))
all_combinations = combinations_1 + combinations_2 + combinations_3


# Main execution
model_list = ["LR", "SVM", "NB", "KNN", "RF", "FCNN", "CNN", "TabPFN", "XGBoost", "LightGBM"]
data = pd.read_csv(os.path.join("..", "result", "00pre-processing", "05final_data-minmax.csv"))
y = data[f"{MODE}-Label"]

for combo in all_combinations:
    print("\n\n\n")
    combined_features = []
    for group in combo:
        combined_features.extend(feature_dict[group])
    feature_group = " + ".join([g for g in combo])
    X = data[combined_features]
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
    
    compare_models(models, X, y, fold=5, plot_name=feature_group, file_path=result_path)
