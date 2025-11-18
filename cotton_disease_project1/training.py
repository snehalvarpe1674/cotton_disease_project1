import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("\n==============================")
print("   TRAINING STARTED")
print("==============================\n")

# Load Dataset
df = pd.read_csv("cotton-crop-disease-dataset (1).csv")
print("Columns:", df.columns.tolist())

# Features and Target
X = df[['Crop', 'Crop Stage']]
y = df['Disease']

# One-Hot Encoding
ct = ColumnTransformer([("encoder", OneHotEncoder(), ['Crop', 'Crop Stage'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Save Encoder
pickle.dump(ct, open("encoder.pkl", "wb"))

# Balance Dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_encoded, y)

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Define Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_model = None
best_accuracy = 0
best_model_name = ""
model_accuracy = {}

print("\n📌 Training Models...\n")

# Train and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_accuracy[name] = acc
    print(f"➡ {name} Accuracy: {acc*100:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save Best Model
pickle.dump(best_model, open("best_model.pkl", "wb"))

print("\n==============================")
print("   TRAINING COMPLETED")
print("==============================")
print(f"✔ Best Model: {best_model_name} with Accuracy: {best_accuracy*100:.2f}%")
print("\n📌 All Model Accuracies:")
for model_name, acc in model_accuracy.items():
    print(f"{model_name}: {acc*100:.2f}%")
print("\n🎉 Model & Encoder Saved Successfully!\n")




