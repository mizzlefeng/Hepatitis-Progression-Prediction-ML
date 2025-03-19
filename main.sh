#!/bin/bash
source activate hep
# After data preprocessing and hyperparameter tuning, perform the following analysis process

# run feature_set_comparison.py
python3 ./scripts/feature_set_comparison.py

# run model_combination_comparison.py
python3 ./scripts/model_combination_comparison.py

# run shap_model.py
python3 ./scripts/shap_model.py

# Web UI
streamlit run web.py