import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("placement_model.pkl")

st.set_page_config(page_title="Placement Predictor", layout="centered")

st.title("🎓 Student Placement Prediction System")
st.write("Enter student details to predict placement chances.")

cgpa = st.slider("CGPA", 5.0, 10.0, 7.5)
communication = st.slider("Communication Skills", 1, 10, 5)
aptitude = st.slider("Aptitude Score", 30, 100, 60)
projects = st.slider("Projects", 0, 6, 2)
internships = st.slider("Internships", 0, 3, 1)
attendance = st.slider("Attendance %", 55, 100, 75)
coding = st.slider("Coding Score", 20, 100, 60)
certifications = st.slider("Certifications", 0, 8, 2)
backlogs = st.slider("Backlogs", 0, 5, 0)
college_tier = st.selectbox("College Tier", [1,2,3])
branch = st.selectbox("Branch", ["CSE","IT","ECE","ME","CE","EE"])

# Branch mapping
branch_map = {"CE":0,"CSE":1,"ECE":2,"EE":3,"IT":4,"ME":5}
branch_encoded = branch_map[branch]

if st.button("Predict Placement"):
    data = np.array([[cgpa, communication, aptitude, projects,
                      internships, attendance, coding,
                      certifications, backlogs, college_tier,
                      branch_encoded]])

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("✅ Likely to be Placed")
    else:
        st.error("❌ Not Likely to be Placed")
