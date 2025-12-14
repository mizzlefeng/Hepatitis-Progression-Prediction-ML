
import streamlit as st
from get_result import predict
from src.common import init_logger

st.set_page_config(page_icon="👩🏻‍⚕", page_title="Hep Predictor", layout="wide")

LOGFILE = f"./log/web.log"

logger, file_handler = init_logger(LOGFILE)

col_center = st.columns([0.1, 0.8, 0.1])[1]
with col_center:
    st.title("Hepatitis Progression Predictor")

    st.caption("""
    This tool uses the XGBoost machine learning model to predict hepatitis progression, 
    analyzing both clinical indicator features and HBV-specific T cell features. 
    By inputting relevant data, the system creates personalized risk assessments for 
    hepatitis progression in the next six months and one year.
    """)

    st.markdown("""
    <style>
    .rainbow-divider {
        height: 3px;
        background: linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet);
        margin: 20px 0;
        border-radius: 2px;
    }
    </style>
    <div class="rainbow-divider"></div>
    """, unsafe_allow_html=True)

    st.header("Input Data")
    selected_features = ['ALT', 'HBV-T', 'HBeAg-T(pH>7)', 'AST', 'HBx-T(pH≤7)', 'AFP']
    respond_info = {feature: None for feature in selected_features}
    
    with st.container(border=True):
        with st.form(key="form1"):
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row2_col1, row2_col2, row2_col3 = st.columns(3)

            with row1_col1:
                respond_info["ALT"] = st.text_input(label="ALT(U/L)")
            with row1_col2:
                respond_info["AST"] = st.text_input(label="AST(U/L)")
            with row1_col3:
                respond_info["AFP"] = st.text_input(label="AFP(ng/ml)")

            with row2_col1:
                respond_info["HBV-T"] = st.text_input(label="HBV-T(4×10⁵ PBMCs)")
            with row2_col2:
                respond_info["HBeAg-T(pH>7)"] = st.text_input(label="HBeAg-T(pl>7)(4×10⁵ PBMCs)")
            with row2_col3:
                respond_info["HBx-T(pH≤7)"] = st.text_input(label="HBx-T(pl≤7)(4×10⁵ PBMCs)")

            form_submitted = st.form_submit_button(label="Predict", icon="🔍️", type="primary")

            def check_form_valid():
                return all(val not in [None, ""] for val in respond_info.values())

        pred_results = None
        if form_submitted:
            if not check_form_valid():
                st.warning("Please fill in all form values!")
                logger.info("Please fill in all form values!")
            else:
                logger.info(respond_info)
                pred_results = predict(respond_info)

        if pred_results is not None:
            prob_dic, waterfall_images = pred_results
            with st.expander("Prediction Results", expanded=True):
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown("<h3 style='text-align: center; color: #336699;'>6-Month Risk Probability</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{prob_dic['6M']}</h2>", unsafe_allow_html=True)
                    st.image(f"data:image/png;base64,{waterfall_images['6M']}", use_container_width=True)
                
                with col2:
                    st.markdown("<h3 style='text-align: center; color: #336699;'>12-Month Risk Probability</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center; color: red;'>{prob_dic['12M']}</h2>", unsafe_allow_html=True)
                    st.image(f"data:image/png;base64,{waterfall_images['12M']}", use_container_width=True)
                
            logger.info(f"6M prediction probability: {prob_dic['6M']}")
            logger.info(f"12M prediction probability: {prob_dic['12M']}")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)