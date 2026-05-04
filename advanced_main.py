import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

# Loading the data
df = pd.read_csv("advanced_student_data.csv")

cols = ['StudyHours','Attendance','PastScore','SleepHours']

# Handling missing data
df = df.drop_duplicates()  # Dropping duplicates
df.replace("None", np.nan, inplace=True)
for col in cols:
    df.fillna({col: df[col].median()}, inplace=True)  # Filling missing values with median
df.fillna({"Internet": df["Internet"].mode()[0]}, inplace=True) # Filling with mode for categorical data

# print("Missing values....")
# print(df.isnull().sum())
# print("Data info....")
# print(df.info())
# print(df.duplicated().sum())

# Encoding categorical data
le = LabelEncoder()
df["Internet"] = le.fit_transform(df["Internet"])  # Yes - 1, No - 0
df["Passed"] = le.fit_transform(df["Passed"])

# Feature scaling/ Normalising
scalar = StandardScaler()
df_scaled = df.copy()
df_scaled[cols] = scalar.fit_transform(df[cols])

x = df_scaled[cols] # Features
y = df_scaled['Passed']  # Target variable

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



