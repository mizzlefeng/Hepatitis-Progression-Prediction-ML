import joblib
import numpy as np
import os
import pandas as pd

# 加载模型和Scaler
model_path = "./model"
scaler_path = "./model/scaler.joblib"
model_list = ["RF", "LightGBM"]
shap_all_features = ['ALT', 'SFU', 'AST', 'AFP', 'DNA load', 'HBeAg1_T', 'IBIL', 'HBpol1_T', 'HBx1_T']
shap_CIF_features = ['ALT', 'AST', 'AFP', 'DNA load', 'IBIL']
shap_STCF_features = ['SFU', 'HBeAg1_T', 'HBpol1_T', 'HBx1_T']
feature_dic = {
    "shap_all": shap_all_features,
    "shap_CIF": shap_CIF_features,
    "shap_STCF": shap_STCF_features,
}
web_map = {
    "ALL": "shap_all",
    "CIF": "shap_CIF",
    "STCF": "shap_STCF"
}
minmax_scaler = joblib.load(scaler_path)
model_dic = {}
for name, features in feature_dic.items():
    for model_name in model_list:
        model_dic[f"{model_name}_{name}"] = joblib.load(os.path.join(model_path, f"{model_name}_{name}.joblib"))

# 输入特征
clinical_features = ['Gender', 'ALT', 'AST', 'Globulin', 'DBIL', 'IBIL', 'AFP', 'DNA load', 'HBsAg', 'HBeAg_COI']
specific_features = ['SFU', 'HBsAg1_T', 'HBsAg2_T', 'HBpol1_T', 'HBpol2_T', 'HBx1_T', 'HBx2_T', 'HBeAg1_T', 'HBeAg2_T']
treat_features = ['ThSched', 'ADV', 'ETV', 'PEG_IFN', 'TAF', 'TDF', 'TFV', 'TMF', 'UnusedD']
full_features = clinical_features + specific_features + treat_features

# 获取用户输入
def get_user_input(features, respond_info):
    """获取用户填写的表单数据并将其转化为模型输入"""
    feature_input = pd.DataFrame(0, index=np.arange(1), columns=full_features).astype(float)
    for feature in features:
        if feature in respond_info:
            feature_input.loc[0, feature] = float(respond_info[feature])
    scaled_features = minmax_scaler.transform(feature_input)
    return scaled_features[:, [full_features.index(f) for f in features]]


# 进行预测
def predict(features_name, respond_info):
    """使用用户输入的特征进行预测"""

    features_name = web_map[features_name]

    features = feature_dic[features_name]
    feature_array = get_user_input(features, respond_info)
    prob_lst = []
    df = pd.DataFrame(feature_array)
    df.columns = features
    for model_name in model_list:
        model = model_dic[f"{model_name}_{features_name}"]
        prediction_prob = model.predict_proba(df)[:, 1][0]
        prob_lst.append(prediction_prob)
    pred_prob = np.mean(prob_lst, axis=0)
    return pred_prob