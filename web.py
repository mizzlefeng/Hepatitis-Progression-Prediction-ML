
import streamlit as st
from src.web_pred import predict
from src.web_log import init_logger

st.set_page_config(page_icon="👩🏻‍⚕", page_title="Hep Predictor", layout="wide")

LOGFILE = f"./log/web.log"

logger,file_handler = init_logger(LOGFILE)

# Title
st.title("Hepatitis Progression Predictor")

# sidebar
st.sidebar.title("Model Type")
add_selectbox = st.sidebar.selectbox(
    "Please select the appropriate model based on your input.",
    ("ALL", "CIF")
)

# Introduction
st.caption("""
This tool analyzes health data, including clinical index features and specific T-cell features, 
to predict hepatitis progression using machine learning ensemble methods. 
By simply inputting relevant data such as :blue[ALT], :blue[SFU], :blue[AST], :blue[AFP], :blue[DNA load], :blue[HBeAg-T(pl>7)], :blue[Indirect bilirubin], :blue[HBpol-T(pl>7)], 
and :blue[HBx-T(pl>7)], the system provides personalized hepatitis progression risk assessments. 
Whether for health monitoring or preventive measures, this platform offers a simple and efficient 
way to help you understand the potential risk of developing hepatitis in the future.""")

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

shap_all_features = ['ALT', 'SFU', 'AST', 'AFP', 'DNA load', 'HBeAg1_T', 'IBIL', 'HBpol1_T', 'HBx1_T']
shap_CIF_features = ['ALT', 'AST', 'AFP', 'DNA load', 'IBIL']
shap_STCF_features = ['SFU', 'HBeAg1_T', 'HBpol1_T', 'HBx1_T']

if add_selectbox == "ALL":
    selected_features = shap_all_features
else:
    selected_features = shap_CIF_features

respond_info = {feature: None for feature in selected_features}

with st.container(border=True):
    with st.form(key="form1"):
        if add_selectbox == "ALL":
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row2_col1, row2_col2, row2_col3 = st.columns(3)
            row3_col1, row3_col2, row3_col3 = st.columns(3)

            with row1_col1:
                respond_info["ALT"] = st.text_input(label="ALT(U/L)")
            with row1_col2:
                respond_info["AST"] = st.text_input(label="AST(U/L)")
            with row1_col3:
                respond_info["AFP"] = st.text_input(label="AFP(ng/ml)")

            with row2_col1:
                respond_info["DNA load"] = st.text_input(label="DNA load(IU/ml)")
            with row2_col2:
                respond_info["IBIL"] = st.text_input(label="Indirect Bilirubin(umol/L)")
            with row2_col3:
                respond_info["SFU"] = st.text_input(label="SFU(4×10⁵ PBMCs)")

            with row3_col1:
                respond_info["HBeAg1_T"] = st.text_input(label="HBeAg-T(pl>7)(4×10⁵ PBMCs)")
            with row3_col2:
                respond_info["HBpol1_T"] = st.text_input(label="HBpol-T(pl>7)(4×10⁵ PBMCs)")
            with row3_col3:
                respond_info["HBx1_T"] = st.text_input(label="HBx-T(pl>7)(4×10⁵ PBMCs)")


        else:
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row2_col1, row2_col2, row2_col3 = st.columns(3)

            with row1_col1:
                respond_info["ALT"] = st.text_input(label="ALT(U/L)")
            with row1_col2:
                respond_info["AST"] = st.text_input(label="AST(U/L)")
            with row1_col3:
                respond_info["AFP"] = st.text_input(label="AFP(ng/ml)")

            with row2_col1:
                respond_info["DNA load"] = st.text_input(label="DNA load(IU/ml)")
            with row2_col2:
                respond_info["IBIL"] = st.text_input(label="Indirect Bilirubin(umol/L)")

        form_submitted = st.form_submit_button(label="Predict",icon="🔍️",type="primary")

        def check_form_valid():
            vals = respond_info.values()
            return all([True if val not in [None, ""] else False for val in vals])

    pred_prob = None
    if form_submitted:
        if not check_form_valid():
            st.warning("Please fill in all form values!")
            logger.info("Please fill in all form values!")
        else:
            print(respond_info)
            logger.info(respond_info)
            pred_prob = predict(add_selectbox,respond_info)

    if pred_prob is not None:
        st.markdown(f"<h4>The probability of developing hepatitis in the next 6 months is <span style='color:red;'>{pred_prob * 100:.2f}%</span></h4>", unsafe_allow_html=True)
        logger.info(f"The probability of developing hepatitis in the next 6 months is {pred_prob * 100:.2f}%.")

# st.image("./main page.webp")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        background-color: #f8f9fa;
        border-top: 1px solid #e9ecef;
        border-left: 1px solid #e9ecef;
        font-family: Arial, sans-serif;
        font-size: 14px;
        color: #6c757d;
    }
    .footer a {
        color: #007bff;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <a href='http://beian.miit.gov.cn/' target='_blank'>苏ICP备2025157471号</a>
    </div>
    """,
    unsafe_allow_html=True
)

import base64
def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )