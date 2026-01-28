import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🚢 Titanic Survival Prediction App")

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# User input
st.header("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
parch = st.number_input("Parents/Children", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)

sex = 1 if sex == "Male" else 0

input_data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare]],
    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
