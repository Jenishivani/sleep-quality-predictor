import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Select required columns
df = df[['Sleep Duration', 'Stress Level', 'Quality of Sleep']]

# Features and target
X = df[['Sleep Duration', 'Stress Level']]
y = df['Quality of Sleep']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully!")