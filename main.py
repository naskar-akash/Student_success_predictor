import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Reading the data
df = pd.read_csv("student_data.csv")

# # Understanding the data
# go to 'understanding_data.txt' for details

# Data preprocessing(encoding)
le = LabelEncoder()
df["Internet"] = le.fit_transform(df["Internet"])  # Yes - 1, No - 0
df["Passed"] = le.fit_transform(df["Passed"])

# print("Encoded dataframe....")
# print(df.head())
# print("Datatypes: ", df.dtypes)

# Feature scalling
features = ['StudyHours','Attendance','PastScore','SleepHours']
scalar = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scalar.fit_transform(df[features])

x = df_scaled[features]
y = df_scaled['Passed']

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test) # Model's prediction 

# Model evaluation
print("Classification Report...")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"])
plt.xlabel("Predicted ---->")
plt.ylabel("Actual ---->")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("----- Predict Your Result -----")
try:
    study_hrs = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance: "))
    past_score = float(input("Enter Past Score: "))
    sleep_hrs = float(input("Enter Sleep Hours: "))

    user_input_df = pd.DataFrame([{
        'StudyHours': study_hrs,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': sleep_hrs
    }])

    user_input_scaled = scalar.fit_transform(user_input_df)
    pred = model.predict(user_input_scaled)[0]

    result = "Pass" if pred == 1 else "Fail"
    print(f"The student may be {result}ed")

except Exception as e:
    print("An error occured: ",e)
