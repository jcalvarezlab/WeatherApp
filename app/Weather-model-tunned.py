import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/weatherAUS_2.csv")

# Use only these 5 features
features = ['Humidity3pm', 'Pressure3pm', 'Sunshine', 'Temperature3pm', 'WindGustSpeed']
X = df[features]
y = df['RainTomorrow'].map({'No': 0, 'Yes': 1})  # assuming your target is categorical

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numeric features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features)
    ]
)

# Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save pipeline
joblib.dump(pipeline, "models/model_pipeline5.pkl")
