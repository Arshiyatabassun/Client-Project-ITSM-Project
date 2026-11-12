from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# 1. Load the trained model
model = joblib.load('ITSM_priority_prediction.joblib')

# 2. Recreate feature list and median/encoder mappings from itsm_data.csv
features = [
    "CI_Name", "CI_Cat", "CI_Subcat", "Status", "Category", 
    "KB_number", "Alert_Status", "No_of_Reassignments", 
    "No_of_Related_Interactions", "No_of_Related_Incidents", 
    "No_of_Related_Changes", "Closure_Code"
]
num_cols = ["No_of_Reassignments", "No_of_Related_Interactions", "No_of_Related_Incidents", "No_of_Related_Changes"]
cat_cols = ["CI_Name", "CI_Cat", "CI_Subcat", "Status", "Category", "KB_number", "Alert_Status", "Closure_Code"]

# Fit encoders and compute medians from original data
df = pd.read_csv("itsm_data.csv")
num_medians = {col: df[col].median() for col in num_cols}

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fill missing values and include 'Unknown'
    series = df[col].fillna("Unknown").astype(str)
    le.fit(series)
    encoders[col] = le

def safe_label_encode(encoder, val):
    val_str = str(val)
    if val_str in encoder.classes_:
        return encoder.transform([val_str])[0]
    elif "Unknown" in encoder.classes_:
        return encoder.transform(["Unknown"])[0]
    else:
        return 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}

        # Process numerical inputs with default median handling
        for col in num_cols:
            raw_val = request.form.get(col, '')
            try:
                input_data[col] = float(raw_val) if raw_val != '' else num_medians[col]
            except ValueError:
                input_data[col] = num_medians[col]

        # Process categorical inputs with safe label encoding
        for col in cat_cols:
            val = request.form.get(col, 'Unknown')
            if not val or val.strip() == '':
                val = 'Unknown'
            input_data[col] = safe_label_encode(encoders[col], val)

        # Build DataFrame in the correct order
        input_df = pd.DataFrame([input_data])[features]

        # Predict
        prediction = model.predict(input_df)[0] + 1

        if prediction == 1:
            result = "Prediction: 1 - Critical"
        elif prediction == 2:
            result = "Prediction: 2 - High"
        elif prediction == 3:
            result = "Prediction: 3 - Medium"
        elif prediction == 4:
            result = "Prediction: 4 - Low"
        else:
            result = "Prediction: 5 - Very Low"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
