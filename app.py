
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model, symptom_list, df = pickle.load(open("medicine_model.pkl", "rb"))

def check_apollo_availability(medicine):
    return f"Available at Apollo Pharmacy (Simulated): {medicine}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = med = dosage = precaution = emergency = lifestyle = diet = availability = ""
    if request.method == 'POST':
        symptoms = request.form['symptoms'].split(',')
        input_vec = pd.DataFrame(np.zeros((1, len(symptom_list))), columns=symptom_list)
        for s in symptoms:
            s = s.strip().capitalize()
            if s in input_vec.columns:
                input_vec[s] = 1
        pred = model.predict(input_vec)[0]
        disease_row = df[df['Disease'] == pred].iloc[0]
        prediction = pred
        med = disease_row['Medicines'] if 'Medicines' in disease_row else "Consult doctor"
        dosage = disease_row['Dosage'] if 'Dosage' in disease_row else "Varies"
        precaution = ", ".join(sym for sym in disease_row.get('Precautions', '').split(',') if sym)
        emergency = disease_row.get('Emergency_Signs', "None")
        lifestyle = disease_row.get('Lifestyle_Tips', "General healthy practices")
        diet = "Eat balanced meals, stay hydrated."
        availability = check_apollo_availability(med)

    return render_template('index.html', prediction=prediction, medicine=med, dosage=dosage,
                           precautions=precaution, emergency=emergency, lifestyle=lifestyle,
                           diet=diet, availability=availability)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_msg = request.json.get('message', '')
    symptom_keywords = [sym for sym in symptom_list if sym.lower() in user_msg.lower()]
    if symptom_keywords:
        return jsonify(response=f"Symptoms recognized: {', '.join(symptom_keywords)}")
    else:
        return jsonify(response="I'm here to help! Please mention your symptoms clearly.")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

