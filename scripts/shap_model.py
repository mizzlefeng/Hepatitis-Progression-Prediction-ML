import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from ..src.common import setup_seed
from ..src.common import get_best_para_from_optuna
from lightgbm import LGBMClassifier
import warnings
import os
import shap
plt.rcParams['font.family'] = 'Arial'
warnings.filterwarnings("ignore")

storage_name = "postgresql://postgres:xxx@127.0.0.1/hep_f1"
setup_seed(42)

def my_lgb(**params):
    return LGBMClassifier(verbose=-1, **params)

model_dict = {
    "RF": RandomForestClassifier,
    "LightGBM": my_lgb
}

result_path = ""

data = pd.read_csv('')
y = data['Label']
clinical_features = ['Gender', 'ALT', 'AST', 'Globulin', 'DBIL', 'IBIL', 'AFP', 'DNA load', 'HBsAg', 'HBeAg_COI']
specific_features = ['SFU', 'HBsAg1_T', 'HBsAg2_T', 'HBpol1_T', 'HBpol2_T', 'HBx1_T', 'HBx2_T', 'HBeAg1_T', 'HBeAg2_T']

feature_dict = {
    "CIF": clinical_features,  # Clinical index features CIF
    "STCF": specific_features,  # Specific T cell features  STCF
}

combined_features = feature_dict['CIF'] + feature_dict['STCF']
X = data[combined_features]

feature_group = "CIF + STCF"

def train_evaluate_models(model_list, X, y):
    models = {}
    for model_name in model_list:
        study_name = feature_group + "_" + model_name
        best_params = get_best_para_from_optuna(study_name, storage_name)
        model_constructor = model_dict[model_name]
        model = model_constructor(**best_params)
        models[model_name] = model
    
    setup_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    shap_values_test_lst = {model_name: [] for model_name in model_list}
    X_test_combined = pd.DataFrame(columns=combined_features)
    y_test_combined = pd.Series(dtype=int)
    for i in range(10):
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_test_combined = pd.concat([X_test_combined, X_test], axis=0)
            y_test_combined = pd.concat([y_test_combined, y_test], axis=0)

            y_pred_prob_dic = {}
            for model_name, model in models.items():
                ml_model = model
                ml_model.fit(X_train, y_train)
                y_pred = ml_model.predict(X_test)
                y_pred_prob = ml_model.predict_proba(X_test)[:, 1]
                y_pred_prob_dic[model_name] = y_pred_prob

                # Calculate SHAP values
                explainer = shap.TreeExplainer(ml_model)
                shap_test_values = explainer.shap_values(X_test)
                shap_values_test_lst[model_name].append(shap_test_values)

    # Calculate mean SHAP values for each model
    shap_test_values = {}

    rf_shap_test_values = np.stack(shap_values_test_lst['RF'], axis=0)[:,:,:,1]
    lgb_shap_test_values = np.stack(shap_values_test_lst['LightGBM'], axis=0)

    stacked_shap_test_values = (rf_shap_test_values + lgb_shap_test_values)/2
    stacked_shap_test_values = stacked_shap_test_values.reshape(-1, 19)
    rf_shap_test_values = rf_shap_test_values.reshape(-1,19)
    lgb_shap_test_values = lgb_shap_test_values.reshape(-1,19)

    return stacked_shap_test_values, rf_shap_test_values, lgb_shap_test_values, X_test_combined, y_test_combined


def calculate_feature_importance(shap_values, features):
    shap_mean = np.mean(np.abs(shap_values), axis=0)
    
    shap_df = pd.DataFrame({
        'feature': features,
        'shap_importance': shap_mean
    })
    shap_df.sort_values(by='shap_importance', ascending=False, inplace=True)
    shap_df['cumulative_shap'] = shap_df.shap_importance.cumsum() / shap_df.shap_importance.sum()

    
    # 95%
    threshold = 0.95
    selected_features = shap_df[shap_df.cumulative_shap < threshold]

    shap_df['shap_importance'] = shap_df['shap_importance']*100
    shap_df['shap_importance'] = shap_df['shap_importance'].apply(lambda x: f"{x:.3f}%")
    shap_df['cumulative_shap'] = shap_df['cumulative_shap']*100
    shap_df['cumulative_shap'] = shap_df['cumulative_shap'].apply(lambda x: f"{x:.3f}%")
    shap_df.to_csv(os.path.join(result_path,"shap_cumsum.csv"), index=False, encoding="utf-8-sig")
    
    return selected_features


model_list = ["RF", "LightGBM"]

combined_shap, rf_shap, lgb_shap, X_test_combined,y_test_combined = train_evaluate_models(model_list, X, y)
selected_features = calculate_feature_importance(combined_shap, X_test_combined.columns.tolist())
print(f"select feature: {selected_features}")

def plot_shap_summary(shap_value,data,name):
    plt.figure(figsize=(5, 8))
    shap.summary_plot(shap_value, data, feature_names=data.columns.tolist(), show=False,cmap="plasma")
    # plt.title(f"SHAP Summary Plot for {name}",c='k')
    plt.ylabel("")
    plt.xlabel("SHAP value",c='k', fontweight='bold', fontsize=14)
    plt.tick_params(axis='x', colors='k')
    plt.tick_params(axis='y', colors='k')
    plt.tick_params(axis='x', labelcolor='k')
    plt.tick_params(axis='y', labelcolor='k')

    fig, ax = plt.gcf(), plt.gca()
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.spines['bottom'].set_linewidth(2)
    ax.yaxis.set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
    # plt.show(block=False)
    plt.savefig(os.path.join(result_path, f'xx.tif'),dpi=300,bbox_inches="tight")
    plt.savefig(os.path.join(result_path, f'xx.pdf'),dpi=600,bbox_inches="tight")



def plot_stacked_shap_summary(shap_values, data, target, title="SHAP Summary Plot (Stacked)"):

    shap_mean = np.mean(np.abs(shap_values), axis=0)

    shap_df = pd.DataFrame({
        'feature': data.columns.tolist(),
        'shap_importance': shap_mean
    })

    shap_values_positive = shap_values[target == 1]
    shap_values_negative = shap_values[target == 0]

    shap_mean_positive = np.mean(np.abs(shap_values_positive), axis=0)
    shap_mean_negative = np.mean(np.abs(shap_values_negative), axis=0)
    
    shap_df_group = pd.DataFrame({
        'feature': data.columns.tolist(),
        'shap_positive': shap_mean_positive,
        'shap_negative': shap_mean_negative
    })

    shap_df_group['shap_positive_ratio'] = shap_df_group['shap_positive'] / (shap_df_group['shap_positive'] + shap_df_group['shap_negative'])
    shap_df_group['shap_negative_ratio'] = shap_df_group['shap_negative'] / (shap_df_group['shap_positive'] + shap_df_group['shap_negative'])

    shap_df['shap_importance_positive'] = shap_df['shap_importance'] * shap_df_group['shap_positive_ratio']
    shap_df['shap_importance_negative'] = shap_df['shap_importance'] * shap_df_group['shap_negative_ratio']

    shap_df['shap_total'] = shap_df['shap_importance_positive'] + shap_df['shap_importance_negative']
    shap_df = shap_df.sort_values(by='shap_total', ascending=True)

    plt.figure(figsize=(8, 9))
    height_bar = 0.7
    plt.barh(shap_df['feature'], shap_df['shap_importance_positive'], color='#5861AC', label='Hepatitis', height=height_bar)
    plt.barh(shap_df['feature'], shap_df['shap_importance_negative'], left=shap_df['shap_importance_negative'], color='#F28080', label='Non-Hepatitis', height=height_bar)

    plt.xlabel('Mean SHAP value', fontsize=15, fontweight='bold',c='k')
    # plt.ylabel('Feature', fontsize=14, fontweight='bold',c='k')
    plt.legend(loc='lower right')
    plt.title("")
    plt.tick_params(axis='x', colors='k')
    plt.tick_params(axis='y', colors='k')
    plt.tick_params(axis='x', labelcolor='k')
    plt.tick_params(axis='y', labelcolor='k')



    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(13)

    plt.subplots_adjust(hspace=0.1, top=0.85)
    ax.margins(y=0.025)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f'xx.tif'),dpi=300,bbox_inches="tight")
    plt.savefig(os.path.join(result_path, f'xx.pdf'),dpi=600,bbox_inches="tight")


plot_shap_summary(combined_shap,X_test_combined,"RF & LightGBM")
plot_stacked_shap_summary(combined_shap, X_test_combined, y_test_combined, "Stacked SHAP Summary Plot (RF + LightGBM)")


