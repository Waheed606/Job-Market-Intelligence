# pyrefly: ignore [missing-import]
import streamlit as st
st.set_page_config(
    page_title="AI Job Market Intelligence",
    layout="wide"
)

st.title("AI Job Market Intelligence System")
st.markdown("Predict Job Roles and Recommend Responsibilities")


st.sidebar.header("Candidate Information")

skills = st.sidebar.text_area(
    "Skills",
    placeholder="Machine Learning, Deep Learning, NLP"
)

keywords = st.sidebar.text_area(
    "Technical Keywords",
    placeholder="Python, TensorFlow, Scikit-learn"
)



experience_level = st.sidebar.selectbox(
    "Experience Level",
    [
        "Entry-Level",
        "Mid-Level",
        "Senior-Level",
        "Lead",
        "Executive"
    ]
)

predict_btn = st.sidebar.button("Predict Job Role")

if predict_btn:

    st.success("Prediction Completed")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👨‍💻 Candidate Profile")

        st.write("Skills:")
        st.write(skills)

        st.write("Keywords:")
        st.write(keywords)

        st.write("Level:")
        st.write(experience_level)

    with col2:
        st.subheader("🎯 Predicted Job Role")

        predicted_role = "ML Engineer"

        st.info(predicted_role)

    st.markdown("---")

    st.subheader("📋 Recommended Responsibilities")

    responsibilities = [
        "Develop machine learning models.",
        "Perform feature engineering.",
        "Train and evaluate deep learning systems.",
        "Deploy ML models into production.",
        "Monitor model performance."
    ]

    for item in responsibilities:
        st.write(f"✅ {item}")