from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open("best_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Load dataset for dropdowns and product mapping
df = pd.read_csv("cotton-crop-disease-dataset (1).csv")
feature_columns = ['Crop', 'Crop Stage']

# Dropdown options
dropdown_values = {col: sorted(df[col].unique()) for col in feature_columns}

# Mapping disease -> recommended product (most frequent if multiple)
product_mapping = df.groupby('Disease')['Recommended Product'].agg(lambda x: x.mode()[0]).to_dict()

# Print which model is loaded
print(f"\n📌 Loaded Model: {type(model).__name__}")

@app.route("/")
def home():
    return render_template("index.html", dropdown_values=dropdown_values)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []
    for col in feature_columns:
        value = request.form.get(col)
        input_data.append(value.strip())

    # Encode input
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    final_input = encoder.transform(input_df)

    # Predict disease
    predicted_disease = model.predict(final_input)[0]

    # Recommended product
    recommended_product = product_mapping.get(predicted_disease.strip(), "No product found")

    return render_template("result.html", predicted_disease=predicted_disease,
                           recommended_product=recommended_product)

if __name__ == "__main__":
    print("🚀 Flask app starting... Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)



