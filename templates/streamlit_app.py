# pyrefly: ignore [missing-import]
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Job Market Intelligence",
    layout="wide"
)

st.title("AI Job Market Intelligence System")
st.markdown("Predict Job Roles and Recommend Responsibilities")


st.sidebar.header("Candidate Information")


keywords = st.sidebar.text_area(
    "Technical Keywords",
    placeholder="Python, TensorFlow, Scikit-learn"
)

experience_level = st.sidebar.selectbox(
    "Experience Level",
    [
        "Fresher",
        "Entry-Level",
        "Junior",
        "Lead",
        "Experienced",
    ]
)

predict_btn = st.sidebar.button("Predict Job Role")

if predict_btn:

    if not keywords.strip():
        st.error("Please enter at least one technical keyword before predicting.")
    else:
        with st.spinner("Calling the model..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "keywords": keywords,
                        "experience_level": experience_level
                    },
                    timeout=15
                )
                response.raise_for_status()
                result_json = response.json()
                predicted_role = result_json["predicted_role"]
                responsibilities_text = result_json["responsibilities"]
                api_error = None
            except requests.exceptions.RequestException as e:
                predicted_role = None
                api_error = str(e)

        if api_error:
            st.error(f"Could not reach the prediction API: {api_error}")
        else:
            st.success("Prediction Completed")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👨‍💻 Candidate Profile")

                st.write("Keywords:")
                st.write(keywords)

                st.write("Level:")
                st.write(experience_level)

            with col2:
                st.subheader("🎯 Predicted Job Role")
                st.info(predicted_role)

            st.markdown("---")

            st.subheader("📋 Recommended Responsibilities")
            st.write(responsibilities_text)