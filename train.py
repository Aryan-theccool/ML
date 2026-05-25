import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)
rows = 10000
data = []
branches = ['CSE','IT','ECE','ME','CE','EE']
tiers = [1,2,3]

for i in range(rows):
    cgpa = round(np.random.uniform(5.0, 10.0),2)
    communication = random.randint(1,10)
    aptitude = random.randint(30,100)
    projects = random.randint(0,6)
    internships = random.randint(0,3)
    attendance = random.randint(55,100)
    coding = random.randint(20,100)
    certifications = random.randint(0,8)
    backlogs = random.randint(0,5)
    tier = random.choice(tiers)
    branch = random.choice(branches)

    score = (
        cgpa*10 +
        communication*4 +
        aptitude*0.4 +
        projects*5 +
        internships*8 +
        attendance*0.2 +
        coding*0.3 +
        certifications*3 -
        backlogs*10 -
        tier*5
    )
    placed = 1 if score > 120 else 0
    data.append([
        cgpa, communication, aptitude, projects,
        internships, attendance, coding,
        certifications, backlogs, tier, branch, placed
    ])

df = pd.DataFrame(data, columns=[
    'CGPA','Communication','Aptitude','Projects',
    'Internships','Attendance','Coding',
    'Certifications','Backlogs','College_Tier',
    'Branch','Placed'
])
df.to_csv("students_large.csv", index=False)

le = LabelEncoder()
df['Branch'] = le.fit_transform(df['Branch'])
X = df.drop("Placed", axis=1)
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Model trained successfully! Accuracy: {acc}")

joblib.dump(model, "placement_model.pkl")
print("Model saved to placement_model.pkl")
