
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

st.set_page_config(page_title="Diabetes Detection App", page_icon="🩺")

st.title("Intelligent Diabetes Detection System")
st.write("This app predicts diabetes risk based on patient vital signs.")
st.write("---")

@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=cols)
    
    X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    return clf

model = get_model()

st.sidebar.header("Patient Information")
name = st.sidebar.text_input("Patient Name", "Fatima")

glucose = st.sidebar.slider("Glucose", 0, 300, 110)

st.sidebar.write("---")
st.sidebar.write("Blood Pressure")
sys_bp = st.sidebar.slider("Systolic BP", 70, 200, 120)
dia_bp = st.sidebar.slider("Diastolic BP", 40, 130, 80)
blood_pressure = sys_bp

st.sidebar.write("---")
st.sidebar.write("Body Measurements")
weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 100.0, 220.0, 165.0)

bmi = weight / ((height / 100) ** 2)
st.sidebar.write(f"Calculated BMI: {bmi:.1f}")

age = st.sidebar.slider("Age", 1, 100, 25)

def create_pdf(p_name, glu, s_bp, d_bp, b_mi, ag, res):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.drawString(100, height - 50, "Medical Report")
    c.drawString(100, height - 90, f"Name: {p_name}")
    c.drawString(100, height - 120, f"Glucose: {glu}")
    c.drawString(100, height - 150, f"Blood Pressure: {s_bp}/{d_bp}")
    c.drawString(100, height - 180, f"BMI: {b_mi:.1f}")
    c.drawString(100, height - 210, f"Age: {ag}")
    c.drawString(100, height - 240, f"Result: {res}")
    
    c.save()
    buffer.seek(0)
    return buffer

if st.button("Detect"):
    data = np.array([[glucose, blood_pressure, bmi, age]])
    pred = model.predict(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pred[0] == 1:
            result = "Positive"
            st.error("Result: Positive (High Risk)")
        else:
            result = "Negative"
            st.success("Result: Negative (Healthy)")
            
        pdf_data = create_pdf(name, glucose, sys_bp, dia_bp, bmi, age, result)
        st.download_button("Download PDF Report", pdf_data, file_name="report.pdf", mime="application/pdf")

    with col2:
        st.write("Comparison Chart:")
        chart_df = pd.DataFrame({
            'Metric': ['Glucose', 'Systolic BP', 'BMI'],
            'Value': [glucose, sys_bp, bmi]
        }).set_index('Metric')
        st.bar_chart(chart_df)
