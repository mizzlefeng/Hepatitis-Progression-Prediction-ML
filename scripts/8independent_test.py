import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from joblib import dump
from src.common import setup_seed
from src.evaluate import evaluate_model, evaluate_model_CI

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Arial'

setup_seed(42)
storage_name = "postgresql://postgres:123...@127.0.0.1/hepatitis"
result_path = "../result/06experiment/"
output_path = "../result/07experiment/"
best_params_path = "../result/06experiment/best_params/"
os.makedirs(best_params_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
data = pd.read_csv('../result/00pre-processing/05final_data-minmax.csv')
val_data = pd.read_csv("../result/00pre-processing/testset/04final_test_data-minmax.csv")

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

config = load_config('./config/ml_config.json')
shap_all_features = ['ALT', 'HBV-T', 'HBeAg-T(pH>7)', 'AST', 'HBx-T(pH≤7)', 'AFP']


def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    ax.plot(thresh_group, net_benefit_model, color = '#025939', label = 'Model', linewidth=2)
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat All', linewidth=2)
    ax.plot((0, 1), (0, 0), color = '#8B511F', linestyle = ':', label = 'Treat None', linewidth=2)

    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = '#025939', alpha = 0.2)

    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Arial', 'fontsize': 14, 'fontweight': 'bold'}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Arial', 'fontsize': 14, 'fontweight': 'bold'}
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='x', colors='k', labelsize=12, width=1.5)
    ax.tick_params(axis='y', colors='k', labelsize=12, width=1.5)

    plt.xticks(fontsize=11, fontweight='bold', c='k')
    plt.yticks(fontsize=11, fontweight='bold', c='k')
    ax.legend(loc = 'upper right',facecolor='none')

    return ax

def load_best_params(shap_feature_name, mode):
    json_file = os.path.join(best_params_path, f"best_params_{shap_feature_name}_{mode}.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params

def cross_validate_trian(mode, n_splits=5, n_repeats=5, val=True):
    setup_seed(42)
    all_results = []
    detail_results = []
    
    best_params = load_best_params("Shap_ALL", mode)
    xgb_params = best_params["xgb_params"]
    
    metrics_all = {"PPV": [], "F1": [], "AUC": [], "AUPR": [], "Gmean": [], "NPV": []}
    y_true_lst = []
    y_prob_lst = []
    y_pred_lst = []
    model_result_dic = {}
    
    if val:
        # Validation mode: cross-validation on training data
        model_result_dic['Model'] = f'XGBoost_{mode}_Val'
        X = data[shap_all_features]
        y = data[f'{mode}-Label']
        
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+repeat)
            for _, (train_index, test_index) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                model = XGBClassifier(**xgb_params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                
                y_true_lst.extend(y_test)
                y_prob_lst.extend(y_pred_prob)
                y_pred_lst.extend(y_pred)
                
                metrics = evaluate_model(y_test, y_pred, y_pred_prob)
                for key in metrics_all.keys():
                    metrics_all[key].append(metrics[key])
        
        metrics_summary = {"Model": 'XGBoost_Train'}
        for metric, values in metrics_all.items():
            metrics_summary[f"{metric}_mean"] = np.mean(values)
            metrics_summary[f"{metric}_std"] = np.std(values)
        
        metrics_all['y_true'] = y_true_lst
        metrics_all['y_prob'] = y_prob_lst
        metrics_all['Model'] = 'XGBoost_Train'
        metrics_CI = evaluate_model_CI(y_true_lst, y_pred_lst, y_prob_lst)
        
        for m in metrics_all.keys():
            for metric, result in metrics_CI.items():
                if m in metric:
                    value_str = f"{np.mean(metrics_all[m]):.3f}({result['95% CI'][0]:.3f}-{result['95% CI'][1]:.3f})"
                    model_result_dic[metric] = value_str
                    new_row = pd.DataFrame([model_result_dic])
                    
        dca_filename = f'XGB_{mode}_Val_DCA'
    
    else:
        # Test mode: train on all data, test on independent set
        model_result_dic['Model'] = f'XGBoost_{mode}_Test'
        
        X_train = data[shap_all_features]
        y_train = data[f'{mode}-Label']
        X_test = val_data[shap_all_features]
        y_test = val_data[f'{mode}-Label']
        
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        dump(model, os.path.join(output_path, f"XGBoost_{mode}.joblib"))
        
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        y_true_lst.extend(y_test)
        y_prob_lst.extend(y_pred_prob)
        y_pred_lst.extend(y_pred)
        
        metrics = evaluate_model(y_test, y_pred, y_pred_prob)
        for key in metrics_all.keys():
            metrics_all[key].append(metrics[key])
        
        metrics_summary = {"Model": 'XGBoost_Test'}
        for metric, values in metrics_all.items():
            metrics_summary[f"{metric}_mean"] = np.mean(values)
            metrics_summary[f"{metric}_std"] = np.std(values)
        
        metrics_all['y_true'] = y_true_lst
        metrics_all['y_prob'] = y_prob_lst
        metrics_all['Model'] = 'XGBoost_Test'
        metrics_CI = evaluate_model_CI(y_true_lst, y_pred_lst, y_prob_lst)
        
        for metric, result in metrics_CI.items():
            value_str = f"{result['value']:.3f}({result['95% CI'][0]:.3f}-{result['95% CI'][1]:.3f})"
            model_result_dic[metric] = value_str
            new_row = pd.DataFrame([model_result_dic])
            
        dca_filename = f'XGB_{mode}_Test_DCA'
    
    # Common code for both modes
    all_results.append(metrics_summary)
    detail_results.append(metrics_all)
    
    # Generate DCA plot
    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_prob_lst, y_true_lst)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_true_lst)
    
    fig, ax = plt.subplots(facecolor='none')
    ax.set_facecolor('none')
    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all)
    plt.tight_layout()
    fig.savefig(os.path.join(output_path, f'{dca_filename}.tif'), format='tif', dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_path, f'{dca_filename}.pdf'), format='pdf', dpi=300, bbox_inches="tight")
    
    return all_results, detail_results, new_row

def main():
    metric_ci_result = pd.DataFrame(columns=['Model', 'F1', 'AUC', 'Gmean', 'AUPR', 'NPV', 'PPV'])
    _, _, ci_row = cross_validate_trian(mode='6M', val=True)
    metric_ci_result = pd.concat([metric_ci_result, ci_row[metric_ci_result.columns]], ignore_index=True)

    _, _, ci_row = cross_validate_trian(mode='6M', val=False)
    metric_ci_result = pd.concat([metric_ci_result, ci_row[metric_ci_result.columns]], ignore_index=True)

    _, _, ci_row = cross_validate_trian(mode='12M', val=True)
    metric_ci_result = pd.concat([metric_ci_result, ci_row[metric_ci_result.columns]], ignore_index=True)

    _, _, ci_row = cross_validate_trian(mode='12M', val=False)
    metric_ci_result = pd.concat([metric_ci_result, ci_row[metric_ci_result.columns]], ignore_index=True)

    metric_ci_result.to_csv(os.path.join(output_path, f"metric_ci_result.csv"), index=False)

    return 0

if __name__ == "__main__":
    main()


