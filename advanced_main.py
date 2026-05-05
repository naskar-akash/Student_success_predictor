import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

x = df_scaled[cols + ["Internet"]] # Features
y = df_scaled['Passed']  # Target variable

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training
model = KNeighborsClassifier()
model.fit(x_train, y_train) 
y_pred = model.predict(x_test) # Model's prediction

# Model evaluation
clf_report = classification_report(y_test, y_pred) # Classification report
cnf_matrix = confusion_matrix(y_test, y_pred) # Confusion matrix

# # Visualising the confusion matrix
# visit 'understanding_data.txt' for details on how to visualise confusion matrix using seaborn heatmap

# Predicting the result for new data
print("----- Predict Your Result -----")
try:
    study_hrs = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance: "))
    past_score = float(input("Enter Past Score: "))
    sleep_hrs = float(input("Enter Sleep Hours: "))
    internet_access = input("Internet Access (Yes/No): ").strip().lower()
    
    internet_access_encoded = 1 if internet_access == "yes" else 0
    
    # Scaling the input features
    user_input_df = pd.DataFrame([{
        'StudyHours': study_hrs,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': sleep_hrs,
        'Internet': internet_access_encoded 
    }])
    user_input_df = user_input_df[cols + ["Internet"]]
    user_input_df[cols] = scalar.transform(user_input_df[cols])

    prediction = model.predict(user_input_df)[0]
    
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Predicted Result: {result}")

except ValueError:
    print("Invalid input. Please enter numeric values for Study Hours, Attendance, Past Score, and Sleep Hours.")
except Exception as e:
    print(f"An error occurred: {e}")
