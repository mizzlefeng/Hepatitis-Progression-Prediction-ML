import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import optuna
from xgboost import XGBClassifier
from src.common import setup_seed, get_best_para_from_optuna, DelongTest
from src.evaluate import evaluate_model
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Arial'

setup_seed(42)
MODE = "6M"
storage_name = "postgresql://postgres:123...@127.0.0.1/hepatitis"
result_path = "../result/06experiment/"
best_params_path = "../result/06experiment/best_params/"
os.makedirs(best_params_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)
data = pd.read_csv('../result/00pre-processing/05final_data-minmax.csv')
y = data[f'{MODE}-Label']

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

config = load_config('./config/ml_config.json')

shap_all_features = ['ALT', 'HBV-T', 'HBeAg-T(pH>7)', 'AST', 'HBx-T(pH≤7)', 'AFP']
shap_CIF_features = ['ALT', 'AST', 'AFP']
shap_STCF_features = ['HBV-T', 'HBeAg-T(pH>7)', 'HBx-T(pH≤7)']
shap_feature_dic = {
    "Shap_ALL": shap_all_features,
    "Shap_CIF": shap_CIF_features,
    "Shap_STCF": shap_STCF_features,
    "NoneReduce": None
}

def optimize_shap_xgboost(X, y, shap_feature_name, study_name, n_trials=100):
    setup_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial):
        try:

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

                model = XGBClassifier(**xgb_params)
                model.fit(X_train, y_train)
                
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
            trial.report(float('-inf'), step=0)
            raise optuna.exceptions.TrialPruned()
    
    try:
        optuna.delete_study(study_name=study_name, storage=storage_name)
    
    except:
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params, study.best_value, study


def save_best_params(shap_feature_name, best_params, best_value):
    params_dict = {
        "shap_feature_name": shap_feature_name,
        "best_value": best_value,
        "best_params": best_params
    }
    xgb_params = {param: best_params[param] for param in config['XGBoost'].keys() if param in best_params}
    
    params_dict["xgb_params"] = xgb_params
    
    output_file = os.path.join(best_params_path, f"best_params_{shap_feature_name}_{MODE}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=2, ensure_ascii=False)
    return params_dict


def optuna_main():
    all_results = {}

    for shap_feature_name, feature_list in shap_feature_dic.items():
        if shap_feature_name != 'NoneReduce':
            study_name = f"{shap_feature_name}_{MODE}"
            X = data[feature_list]
            y = data[f'{MODE}-Label']
            best_params, best_value, _ = optimize_shap_xgboost(
                X, y, shap_feature_name, study_name, n_trials=200
            )
            
            if best_params is not None and best_value is not None:
                params_dict = save_best_params(shap_feature_name, best_params, best_value)
                all_results[shap_feature_name] = params_dict     
                
    all_results_file = os.path.join(best_params_path, f"all_shap_results_{MODE}.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

def load_best_params(shap_feature_name, mode):
    json_file = os.path.join(best_params_path, f"best_params_{shap_feature_name}_{mode}.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        params = json.load(f)

    return params

def load_data():
    data = pd.read_csv('../result/00pre-processing/05final_data-minmax.csv')
    y = data[f'{MODE}-Label']
    clinical_features = ['Gender', 'ALT', 'AST', 'Albumin', 'GGT', 'DBIL', 'IBIL', 'AFP', 'DNA load', 'HBsAg']
    specific_features = ['HBV-T', 'HBsAg-T(pH>7)', 'HBsAg-T(pH≤7)', 'HBpol-T(pH>7)', 'HBpol-T(pH≤7)', 
                        'HBx-T(pH>7)', 'HBx-T(pH≤7)', 'HBeAg-T(pH>7)', 'HBeAg-T(pH≤7)']
    combined_features = clinical_features + specific_features
    X = data[combined_features]
    return X, y

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

def cross_validate_shap_with_best_params(n_splits=5, n_repeats=5):
    all_results = []
    detail_results = []
    
    for shap_feature_name, feature_list in shap_feature_dic.items():
        if shap_feature_name != 'NoneReduce':
            best_params = load_best_params(shap_feature_name, MODE)
            X = data[feature_list]
            y = data[f'{MODE}-Label']
            xgb_params = best_params["xgb_params"]
    
            metrics_all = {"PPV": [], "F1": [], "AUC": [], "AUPR": [], "Gmean": [], "NPV": []}
        
            y_true_lst = []
            y_prob_lst = []
            for repeat in range(n_repeats):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+repeat)
                
                for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    y_true_lst.extend(y_test)

                    model = XGBClassifier(**xgb_params)
                    model.fit(X_train, y_train)
                
                    y_pred = model.predict(X_test)
                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                    y_prob_lst.extend(y_pred_prob)
                    
                    metrics = evaluate_model(y_test, y_pred, y_pred_prob)
                    for key in metrics_all.keys():
                        metrics_all[key].append(metrics[key])
                    
            metrics_summary = {"Model": f'{shap_feature_name}'}
            for metric, values in metrics_all.items():
                metrics_summary[f"{metric}_mean"] = np.mean(values)
                metrics_summary[f"{metric}_std"] = np.std(values)
            metrics_all['y_true'] = y_true_lst
            metrics_all['y_prob'] = y_prob_lst
            metrics_all['Model'] = f'{shap_feature_name}'
        
            all_results.append(metrics_summary)
            detail_results.append(metrics_all)
        
        else:
            feature_group = "CIF + STCF"
            model_name = "XGBoost"
            study_name = f"{feature_group}_{model_name}_{MODE}"
            storage_name = "postgresql://postgres:123...@127.0.0.1/hepatitis"
            xgb_params = get_best_para_from_optuna(study_name, storage_name)
            X,y = load_data()
    
            metrics_all = {"PPV": [], "F1": [], "AUC": [], "AUPR": [], "Gmean": [], "NPV": []}

            y_true_lst = []
            y_prob_lst = []
            for repeat in range(n_repeats):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42+repeat)
                
                for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    y_true_lst.extend(y_test)

                    model = XGBClassifier(**xgb_params)
                    model.fit(X_train, y_train)
                
                    y_pred = model.predict(X_test)
                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                    y_prob_lst.extend(y_pred_prob)
                    
                    metrics = evaluate_model(y_test, y_pred, y_pred_prob)
                    for key in metrics_all.keys():
                        metrics_all[key].append(metrics[key])
                    
            metrics_summary = {"Model": f'XGBoost'}
            for metric, values in metrics_all.items():
                metrics_summary[f"{metric}_mean"] = np.mean(values)
                metrics_summary[f"{metric}_std"] = np.std(values)
            metrics_all['y_true'] = y_true_lst
            metrics_all['y_prob'] = y_prob_lst
            metrics_all['Model'] = f'XGBoost'
        
            all_results.append(metrics_summary)
            detail_results.append(metrics_all)
    
    return all_results, detail_results


def cv_compare_main():
    results, detail_results = cross_validate_shap_with_best_params(n_splits=5, n_repeats=5)

    save_results_to_excel_annotate(results, os.path.join(result_path, f"Shap_{MODE}_annotate.csv"))
    save_results_to_excel_raw(results, os.path.join(result_path, f"Shap_{MODE}_raw.csv"))
    save_results_to_excel_detail(detail_results, os.path.join(result_path, f"Shap_{MODE}_detail.pkl"))


def get_delong_result():
    df = pd.read_pickle(os.path.join(result_path, f"Shap_{MODE}_detail.pkl"))

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
    result_df.to_csv(os.path.join(result_path, f'Shap_{MODE}_delong.csv'),index=False)
    return result_df

if __name__ == "__main__":
    optuna_main()
    cv_compare_main()
    get_delong_result()








