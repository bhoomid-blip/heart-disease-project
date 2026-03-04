import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# load data
df = pd.read_csv("heart_v2 (2).csv")

# split features and target
X = df.drop("heart disease", axis=1)
y = df["heart disease"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42
)

# train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("heart_model.pkl", "wb"))

print("✅ Model trained and saved!")