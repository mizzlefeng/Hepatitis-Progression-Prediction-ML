import os
import json
import warnings
from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import optuna
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from src.common import init_logger, setup_seed, get_best_para_from_optuna, DelongTest
from src.evaluate import evaluate_model
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Arial'

setup_seed(42)
MODE = "6M"
storage_name = "postgresql://postgres:123...@127.0.0.1/hepatitis"
result_path = "../result/04experiment/"
best_params_path = "../result/04experiment/best_params/"
os.makedirs(best_params_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

config = load_config('./config/ml_config.json')

LOGFILE = os.path.join(best_params_path, f"imbalanced_xgboost_optimization_{MODE}.log")
if os.path.exists(LOGFILE):
    os.remove(LOGFILE)
logger, file_handler = init_logger(LOGFILE)

sampling_methods = {
    "SMOTE": SMOTE,
    "ADASYN": ADASYN,
    "NearMiss": NearMiss,
    "NoneBalance": ''
}

# Parameter space for different sampling methods
sampling_params_space = {
    "SMOTE": {
        "sampling_strategy": {"type": "suggest_categorical", "args": [["auto", 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]]},
        "k_neighbors": {"type": "suggest_int", "args": [3, 10]},
        "random_state": {"type": "suggest_int", "args": [42, 42]}
    },
    "ADASYN": {
        "sampling_strategy": {"type": "suggest_categorical", "args": [["auto", 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]]},
        "n_neighbors": {"type": "suggest_int", "args": [3, 10]},
        "random_state": {"type": "suggest_int", "args": [42, 42]}
    },
    "NearMiss": {
        "version": {"type": "suggest_int", "args": [1, 3]},
        "n_neighbors": {"type": "suggest_int", "args": [3, 10]},
    }
}

def load_data():
    data = pd.read_csv('../result/00pre-processing/05final_data-minmax.csv')
    y = data[f'{MODE}-Label']
    clinical_features = ['Gender', 'ALT', 'AST', 'Albumin', 'GGT', 'DBIL', 'IBIL', 'AFP', 'DNA load', 'HBsAg']
    specific_features = ['HBV-T', 'HBsAg-T(pH>7)', 'HBsAg-T(pH≤7)', 'HBpol-T(pH>7)', 'HBpol-T(pH≤7)',
                        'HBx-T(pH>7)', 'HBx-T(pH≤7)', 'HBeAg-T(pH>7)', 'HBeAg-T(pH≤7)']
    combined_features = clinical_features + specific_features
    X = data[combined_features]
    return X, y

def load_best_params(method_name, mode):
    json_file = os.path.join(best_params_path, f"best_params_{method_name}_{mode}.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params

def optimize_imbalanced_xgboost(X, y, sampling_method_name, study_name, n_trials=100):
    setup_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sampling_method_class = sampling_methods[sampling_method_name]
    
    def objective(trial):
        try:
            sampling_params = {}
            for param_name, param_info in sampling_params_space[sampling_method_name].items():
                param_method = getattr(trial, param_info["type"])
                sampling_params[param_name] = param_method(param_name, *param_info["args"])
            
            xgb_params = {}
            for param_name, param_info in config['XGBoost'].items():
                param_method = getattr(trial, param_info["type"])
                xgb_params[param_name] = param_method(param_name, *param_info["args"])
            
            f1_list = []
            roc_auc_list = []
            aupr_list = []
            
            for _, (train_index, test_index) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                sampler = sampling_method_class(**sampling_params)
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

                model = XGBClassifier(**xgb_params)
                model.fit(X_resampled, y_resampled)
                
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_prob)
                aupr = average_precision_score(y_test, y_pred_prob)
                
                f1_list.append(f1)
                roc_auc_list.append(roc_auc)
                aupr_list.append(aupr)
            
            return np.mean(f1_list)
        except Exception as e:
            logger.error(f"Optimization process error: {str(e)}")
            trial.report(float('-inf'), step=0)
            raise optuna.exceptions.TrialPruned()
    
    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    except:
        pass
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params, study.best_value, study

def cross_validate_imbalanced_methods_with_best_params(X, y, n_splits=5, n_repeats=5):
    all_results = []
    detail_results = []
    
    for method_name, method_class in sampling_methods.items():
        metrics_all = {"PPV": [], "F1": [], "AUC": [], "AUPR": [], "Gmean": [], "NPV": []}
        y_true_lst = []
        y_prob_lst = []
        
        # Get parameters based on method type
        if method_name != 'NoneBalance':
            best_params = load_best_params(method_name, MODE)
            sampling_params = best_params["sampling_params"]
            xgb_params = best_params["xgb_params"]
            model_label = f'{method_name}+XGBoost'
        else:
            feature_group = "CIF + STCF"
            model_name = "XGBoost"
            study_name = f"{feature_group}_{model_name}_{MODE}"
            xgb_params = get_best_para_from_optuna(study_name, storage_name)
            model_label = "XGBoost"
        
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+repeat)
            
            for _, (train_index, test_index) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                y_true_lst.extend(y_test)
                
                # Apply sampling if method is not NoneBalance
                if method_name != 'NoneBalance':
                    sampler = method_class(**sampling_params)
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                    model = XGBClassifier(**xgb_params)
                    model.fit(X_resampled, y_resampled)
                else:
                    model = XGBClassifier(**xgb_params)
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                y_prob_lst.extend(y_pred_prob)
                
                metrics = evaluate_model(y_test, y_pred, y_pred_prob)
                for key in metrics_all.keys():
                    metrics_all[key].append(metrics[key])
        
        # Calculate summary metrics
        metrics_summary = {"Model": model_label}
        for metric, values in metrics_all.items():
            metrics_summary[f"{metric}_mean"] = np.mean(values)
            metrics_summary[f"{metric}_std"] = np.std(values)
        
        metrics_all['y_true'] = y_true_lst
        metrics_all['y_prob'] = y_prob_lst
        metrics_all['Model'] = model_label
        
        all_results.append(metrics_summary)
        detail_results.append(metrics_all)
    
    return all_results, detail_results

# Save best parameters
def save_best_params(sampling_method_name, best_params, best_value):
    params_dict = {
        "sampling_method": sampling_method_name,
        "best_value": best_value,
        "best_params": best_params
    }
    
    sampling_params = {param: best_params[param] for param in sampling_params_space[sampling_method_name].keys() if param in best_params}
    xgb_params = {param: best_params[param] for param in config['XGBoost'].keys() if param in best_params}
    
    params_dict["sampling_params"] = sampling_params
    params_dict["xgb_params"] = xgb_params
    
    output_file = os.path.join(best_params_path, f"best_params_{sampling_method_name}_{MODE}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Best parameters saved to: {output_file}")
    return params_dict

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

    mean_cols = ['Model','F1_mean','AUC_mean','Gmean_mean','AUPR_mean','NPV_mean','PPV_mean']
    std_cols = ['Model','F1_std','AUC_std','Gmean_std','AUPR_std','NPV_std','PPV_std']
    need_col = ['Model','F1','AUC','Gmean','AUPR','NPV','PPV']
    df_mean = df[mean_cols]
    df_mean_transposed = annotate_top_values(df_mean)
    df_std = df[std_cols]
    df_combined = pd.DataFrame()
    for col in need_col:
        if col == 'Model':
            df_combined[col] = df[col]
        else:
            df_combined[col] = [f"{m}±{s:.3f}" for m, s in zip(df_mean_transposed[f"{col}_mean"], df_std[f"{col}_std"])]
    
    df_combined.to_csv(filename, index=False, encoding='utf-8-sig')

def save_results_to_excel_raw(results, filename):
    df = pd.DataFrame(results)

    mean_cols = ['Model','F1_mean','AUC_mean','Gmean_mean','AUPR_mean','NPV_mean','PPV_mean']
    need_col = ['Model','F1','AUC','Gmean','AUPR','NPV','PPV']
    df_mean = df[mean_cols]
    df_combined = pd.DataFrame()
    for col in need_col:
        if col == 'Model':
            df_combined[col] = df[col]
        else:
            df_combined[col] = [f"{m:.4f}" for m in df_mean[f"{col}_mean"]]
    
    df_combined.to_csv(filename, index=False, encoding='utf-8-sig')

def save_results_to_excel_detail(results, filename):
    df = pd.DataFrame(results)
    df.to_pickle(filename)

def optuna_main():
    X, y = load_data()
    logger.info(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    all_results = {}
    for sampling_method_name in sampling_methods.keys():
        if sampling_method_name != 'NoneBalance':
            logger.info(f"\nStarting optimization for {sampling_method_name} ......")
            
            study_name = f"{sampling_method_name}+XGBoost_{MODE}"
            best_params, best_value, _ = optimize_imbalanced_xgboost(
                X, y, sampling_method_name, study_name, n_trials=500
            )
            
            if best_params is not None and best_value is not None:
                logger.info(f"{sampling_method_name} best F1 score: {best_value:.4f}")
                params_dict = save_best_params(sampling_method_name, best_params, best_value)
                all_results[sampling_method_name] = params_dict
            else:
                logger.error(f"{sampling_method_name} optimization failed")
    
    # Save all results to a file
    all_results_file = os.path.join(best_params_path, f"all_sampling_methods_results_{MODE}.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nAll results saved to: {all_results_file}")
    logger.info("Imbalanced data processing methods and XGBoost parameter optimization completed")
    
    file_handler.close()


def cv_compare_main():
    X, y = load_data()
    results, detail_results = cross_validate_imbalanced_methods_with_best_params(X, y, n_splits=5, n_repeats=5)

    save_results_to_excel_annotate(results, os.path.join(result_path, f"imbalanced_methods_{MODE}_annotate.csv"))
    save_results_to_excel_raw(results, os.path.join(result_path, f"imbalanced_methods_{MODE}_raw.csv"))
    save_results_to_excel_detail(detail_results, os.path.join(result_path, f"imbalanced_methods_{MODE}_detail.pkl"))

def radar_main():
    fig, ax = plt.subplots(figsize=(8, 8), 
                          subplot_kw={'polar': True}, 
                          constrained_layout=True,facecolor='none')
    ax.set_facecolor('none')
    metric_radar = ["AUC", "AUPR", "Gmean", "F1", "PPV", "NPV"]
    plt.rcParams['font.family'] = 'Arial'
    colors = sns.color_palette("tab10")

    all_handles = []
    all_labels = []

    base_output_file = os.path.join(result_path, f"imbalanced_methods_{MODE}_raw.csv")
    result_df = pd.read_csv(base_output_file)

    angles = [n / float(len(metric_radar)) * 2 * pi for n in range(len(metric_radar))]
    angles += angles[:1]

    for idx, combo in enumerate(result_df['Model'].unique()):
        subset = result_df[result_df['Model'] == combo]
        values = subset[metric_radar].values.flatten().tolist()
        values += values[:1]
        
        line = ax.plot(angles, values, color=colors[idx], linewidth=1.5, 
                        linestyle='solid', marker='o', markersize=3, label=combo)
        ax.fill(angles, values, color=colors[idx], alpha=0.15)

        if combo not in all_labels:
            all_handles.append(line[0])
            all_labels.append(combo)

    ax.text(0.5, 1.1, f"{MODE} Imbalanced Methods Performance Comparison", transform=ax.transAxes, 
            fontsize=16, fontweight='bold', ha='center')
    
    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    
    ax.set_xticklabels([])
    for i, (metric, angle) in enumerate(zip(metric_radar, angles[:-1])):
        if i == 0:
            ha = 'center'
            va = 'bottom'
            xytext = (0, 10)
        elif i == 1:
            ha = 'left'
            va = 'bottom'
            xytext = (8, 5)
        elif i == 2:
            ha = 'left'
            va = 'top'
            xytext = (8, -5)
        elif i == 3:
            ha = 'center'
            va = 'top'
            xytext = (0, -10)
        elif i == 4:
            ha = 'right'
            va = 'top'
            xytext = (-8, -5)
        elif i == 5:
            ha = 'right'
            va = 'bottom'
            xytext = (-8, 5)
        else:
            ha = 'center'
            va = 'center'
            xytext = (0, 0)

        ax.annotate(metric, 
                    xy=(angle, 1),
                    xytext=xytext,
                    textcoords='offset points',
                    ha=ha,
                    va=va,
                    fontsize=14,
                    fontweight='bold',
                    color='black')

    ax.set_rlabel_position(30)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], 
                        color="grey", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    ax.spines['polar'].set_linewidth(2)
    for child in ax.get_yticklabels():
        child.set_fontweight('bold')

    legend = ax.legend(all_handles, all_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.18),
              prop={'weight': 'bold', 'size': 13}, 
              ncol=min(2, len(all_labels)),
              frameon=True, 
              framealpha=0.5,
              edgecolor='black',
              markerfirst=True)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'radar_plot_{MODE}.tif'), format='tif', dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(result_path, f'radar_plot_{MODE}.pdf'), format='pdf', dpi=300, bbox_inches="tight")

def get_delong_result():

    df = pd.read_pickle(os.path.join(result_path, f"imbalanced_methods_{MODE}_detail.pkl"))

    df['y_true_tuple'] = df['y_true'].apply(tuple)
    assert df['y_true_tuple'].nunique() == 1, "Multiple unique values found in 'y_true_tuple'"

    models = df["Model"].unique()
    results = []

    y_label = df['y_true'][0]
    baseline_model = "XGBoost"
    if baseline_model in models:
        other_models = [model for model in models if model != baseline_model]
        for model2 in other_models:
            pred1 = df[df["Model"] == baseline_model]["y_prob"].values[0]
            pred2 = df[df["Model"] == model2]["y_prob"].values[0]
            
            delong_test = DelongTest(preds1=pred1, preds2=pred2, label=y_label)
            results.append((baseline_model, model2, delong_test.z, delong_test.p))

    result_df = pd.DataFrame(results, columns=["Model1", "Model2", "Z","P_value"])
    result_df.to_csv(os.path.join(result_path, f'imbalanced_{MODE}_delong.csv'),index=False)
    return result_df

if __name__ == "__main__":
    optuna_main()
    cv_compare_main()
    radar_main()
    get_delong_result()




