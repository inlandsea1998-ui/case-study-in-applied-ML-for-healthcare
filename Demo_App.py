import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix


# -----------------------------
# Load and train model
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Train tuned SVM (replace with your best params if different)
model = SVC(probability=True, kernel='rbf', C=50, random_state=24)
model.fit(X_train, y_train)

# Train MLP model
mlp_model = MLPClassifier(random_state=24, max_iter=1000)
mlp_model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Breast Cancer Prediction Demo")

st.markdown("""
### About This Demo
This demo uses the **Breast Cancer Wisconsin dataset** to classify tumors as malignant or benign.  
You can adjust feature values below to see how the model responds.  
The app also shows evaluation metrics like ROC curve and confusion matrix to illustrate performance.
""")

# -----------------------------
# Input section
# -----------------------------
model_choice = st.selectbox("Choose a model:", ["SVM", "Neural Network"])
st.header("Input Tumor Features")

inputs = []
for i, feature in enumerate(data.feature_names):
    val = st.number_input(
        f"{feature}",
        float(X[:, i].min()), 
        float(X[:, i].max()), 
        float(X[:, i].mean())  # default value
    )
    inputs.append(val)
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    
    if model_choice == "SVM":
        prediction = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][prediction]
    else:
        prediction = mlp_model.predict(input_array)[0]
        prob = mlp_model.predict_proba(input_array)[0][prediction]
    
    result = "Malignant" if prediction == 0 else "Benign"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {prob:.2f}")

# -----------------------------
# Evaluation Tabs
# -----------------------------
st.header("Model Evaluation")
tab1, tab2 = st.tabs(["SVM Evaluation", "Neural Network Evaluation"])

# SVM evaluation
with tab1:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    st.write("Confusion Matrix (SVM)")
    st.write(cm)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1],[0,1],'r--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# Neural Network evaluation
with tab2:
    y_pred = mlp_model.predict(X_test)
    y_prob = mlp_model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    
    st.write("Confusion Matrix (Neural Network)")
    st.write(cm)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1],[0,1],'r--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)