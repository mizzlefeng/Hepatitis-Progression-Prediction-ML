import joblib
import numpy as np
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64

model_path = "../web/model"
scaler_path = "../web/model/scaler.joblib"
shap_all_features = ['ALT', 'HBV-T', 'HBeAg-T(pH>7)', 'AST', 'HBx-T(pH≤7)', 'AFP']

mode_list = ["6M", "12M"]
minmax_scaler = joblib.load(scaler_path)

clinical_features = ['Gender', 'ALT', 'AST', 'Albumin', 'GGT', 'DBIL', 'IBIL', 'AFP', 'DNA load', 'HBsAg']
specific_features = ['HBV-T', 'HBsAg-T(pH>7)', 'HBsAg-T(pH≤7)', 'HBpol-T(pH>7)', 'HBpol-T(pH≤7)', 'HBx-T(pH>7)', 'HBx-T(pH≤7)', 'HBeAg-T(pH>7)', 'HBeAg-T(pH≤7)']
treat_features = ['ThSched', 'ADV', 'ETV', 'PEG-IFN', 'TAF', 'TDF', 'TFV', 'TMF']
full_features = clinical_features + specific_features + treat_features

def get_original_scale_explanation(expl_main):
    alt_index_scaler = list(minmax_scaler.feature_names_in_)
    alt_index_expl = list(expl_main.feature_names)
    
    feature_indices = [alt_index_scaler.index(f) for f in alt_index_expl]
    data_normalized = expl_main.data.reshape(1, -1)
    
    if hasattr(minmax_scaler, 'mean_'):
        # StandardScaler
        scale = minmax_scaler.scale_[feature_indices].reshape(1, -1)
        mean = minmax_scaler.mean_[feature_indices].reshape(1, -1)
        data_original = data_normalized * scale + mean
    else:
        # MinMaxScaler
        data_min = minmax_scaler.data_min_[feature_indices].reshape(1, -1)
        data_max = minmax_scaler.data_max_[feature_indices].reshape(1, -1)
        data_original = data_normalized * (data_max - data_min) + data_min
    
    expl_original = shap.Explanation(
        values=expl_main.values,
        data=data_original[0],
        feature_names=expl_main.feature_names,
        base_values=expl_main.base_values,
        output_names=expl_main.output_names if hasattr(expl_main, 'output_names') else None
    )
    return expl_original

def plot_waterfall(expl_main, rp="./"):

    plt.figure()
    shap.plots.waterfall(expl_main, show=False)
    plt.gcf().set_size_inches(7,7.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(rp,f'Waterfall_{MODE}.tif'),dpi=300,bbox_inches="tight")
    # plt.savefig(os.path.join(rp,f'Waterfall_{MODE}.pdf'),dpi=600,bbox_inches="tight")

# 获取用户输入
def get_user_input(features, respond_info):

    feature_input = pd.DataFrame(0, index=np.arange(1), columns=full_features).astype(float)
    for feature in features:
        if feature in respond_info:
            feature_input.loc[0, feature] = float(respond_info[feature])
    scaled_features = minmax_scaler.transform(feature_input)
    return scaled_features[:, [full_features.index(f) for f in features]]


# 进行预测
def predict(respond_info):

    feature_array = get_user_input(shap_all_features, respond_info)
    prob_dic = {}
    waterfall_images = {}
    df = pd.DataFrame(feature_array, columns=shap_all_features)
    
    for mode in mode_list:
        model = joblib.load(os.path.join(model_path, f"XGBoost_{mode}.joblib"))
        prediction_prob = model.predict_proba(df)[:, 1][0]
        prediction_prob = prediction_prob*100
        prob_dic[mode] = f'{prediction_prob:.4f}%'
        print(f"{mode} prob: {prediction_prob:.4f}%")

        explainer = shap.Explainer(model)
        shap_values = explainer(df)
        sample_expl = shap_values[0]
        original_expl = get_original_scale_explanation(sample_expl)
        
        buffer = io.BytesIO()
        plt.figure()
        shap.plots.waterfall(original_expl, show=False)
        plt.gcf().set_size_inches(7, 7.5)
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()

        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        waterfall_images[mode] = image_base64
        
    return prob_dic, waterfall_images
