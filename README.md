🩺 AI Health Advisor An AI-powered Health Advisor web app that predicts possible diseases based on user-reported symptoms. The system also recommends personalized medicines, precautions, emergency signs, and lifestyle tips. Includes an integrated chatbot symptom checker for an interactive health consultation experience.

📌 Features ✅ Disease prediction from symptoms using ML

💊 Personalized medicine recommendations

🛡️ Health precautions and emergency signs

🌱 Lifestyle and diet suggestions

🤖 Chatbot symptom checker

🖥️ Flask-based web interface

📊 99.8% model accuracy

📁 Dataset This project uses four curated datasets:

dataset.csv: Main dataset with diseases, symptoms, medicines, dosages, etc.

symptom_Description.csv: Descriptions for each symptom

symptom_precaution.csv: Precautionary measures for symptoms

Symptom-severity.csv: Severity ranking for each symptom

🚀 How It Works Users input symptoms via form or chatbot

ML model predicts the most probable disease

System fetches corresponding medicines, precautions, and tips

Output is displayed in a clean, tabbed interface

🛠️ Tech Stack Python

Scikit-learn

Pandas / NumPy

Flask (Backend)

HTML + CSS (Frontend)

Google Colab (Model Training)

🧠 Model Accuracy Achieved 99.80% accuracy using a Random Forest Classifier trained on a cleaned, symptom-encoded dataset.

🤖 Chatbot The chatbot allows users to describe their symptoms in natural language, which gets parsed into structured symptom inputs for the model.

📷 Screenshots Add screenshots here after running the app to show the user interface and features.

💡 Future Improvements Add multilingual support for wider accessibility

Integrate with pharmacy APIs like Apollo for live medicine availability

Enable user account management for history tracking
