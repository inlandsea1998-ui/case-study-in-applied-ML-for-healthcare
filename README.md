# Breast Cancer Classification with SVM and Neural Networks

## 📌 Project Overview
Breast cancer is one of the most common cancers worldwide, and early detection is critical for improving patient outcomes.  
This project applies **machine learning models** to the **Breast Cancer Wisconsin dataset** to classify tumors as **malignant** or **benign**.  

I implemented and compared a **Support Vector Machine (SVM)** with hyperparameter tuning and a **Multi-Layer Perceptron (MLP) neural network**, evaluated their performance with accuracy, confusion matrices, and ROC curves, and discussed the implications of false positives and false negatives in a medical context.

I chose SVM for its robustness on smaller datasets and MLP to test a flexible non-linear model. I used GridSearchCV to tune hyperparameters, ensuring the models were optimized rather than relying on defaults.

I built a **[Web app](https://case-study-in-applied-ml-for-healthcare.streamlit.app/)** to make this project interactive.
Instead of just reading code, you can enter tumor feature values and see real‑time predictions from both the SVM and Neural Network models.

---

## 📊 Dataset
- **Source:** Scikit-learn’s `load_breast_cancer` dataset  
- **Samples:** 569  
- **Features:** 30 numeric features describing cell nuclei (e.g., radius, texture, smoothness)  
- **Target:** Binary classification (malignant vs. benign)  

---

## ⚙️ Methodology
1. **Data Preparation**
   - Train-test split (80/20)  
   - Standardization (optional depending on model)  

2. **Models**
   - **SVM** with hyperparameter tuning using GridSearchCV  
   - **MLPClassifier** (Neural Network)  

3. **Evaluation Metrics**
   - Accuracy  
   - Confusion Matrix  
   - ROC Curve & AUC  

---

## ✅ Results
- **Best SVM Model**
  - Kernel: RBF  
  - C: 50  
  - Accuracy: ~97%  
  - AUC: ~0.99  

- **Neural Network Model**
  - Accuracy: ~95%  
  - AUC: ~0.98  

- **Confusion Matrix Insights**
  - ROC shows the trade-off between sensitivity (true positive rate) and specificity (false positive rate)
  - AUC close to 1.0 indicates excellent discrimination ability.
  - The confusion matrix revealed very few false negatives, which is critical in cancer detection.
  - The ROC curve showed an AUC of 0.99, meaning the model is highly effective at distinguishing malignant from benign cases.

---

## 🔍 Key Insights
- SVM slightly outperformed the neural network, likely due to the dataset’s relatively small size and well-separated classes.  
- ROC curves showed both models are highly effective, but SVM’s margin-based approach gave it a slight edge.  
- In real-world applications, minimizing **false negatives** is more important than maximizing overall accuracy.  

---

## 🚀 Extensions
- **Deployment demo** using Streamlit for interactive predictions. The app includes:
   - Default example inputs (so you can test quickly)
   - ROC curve and confusion matrix visualizations
   - A short “About” section explaining the dataset and purpose

---

## 🏁 Conclusion
This project demonstrates the ability to:  
- Apply machine learning algorithms to real-world health data  
- Tune hyperparameters and evaluate models rigorously  
- Interpret results in a domain-specific context (medical diagnostics)  

