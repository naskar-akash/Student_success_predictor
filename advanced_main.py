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

x = df_scaled[cols] # Features
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

# Visualising the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"])
plt.xlabel("Predicted ---->")
plt.ylabel("Actual ---->")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("Figure/confusion_matrix.png", dpi=300, bbox_inches='tight') # Saving the confusion matrix as an image
plt.show()

