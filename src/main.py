import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("datasets/adult.csv")

# Ubah tanda "?" jadi NaN
cols = ["workclass", "occupation", "native-country"]
for c in cols:
    df[c] = df[c].replace("?", np.nan)

# Drop data yang null
df.dropna(inplace=True)

# =========================
# 2. Penyederhanaan kategori
# =========================
replace_workclass = {
    "Private": "Private",
    "Self-emp-not-inc": "Self-Employed",
    "Self-emp-inc": "Self-Employed",
    "Local-gov": "Government",
    "State-gov": "Government",
    "Federal-gov": "Government",
    "Without-pay": "Unemployed",
    "Never-worked": "Unemployed"
}
df["workclass"] = df["workclass"].replace(replace_workclass)

replace_education = {
    "Preschool": "Low",
    "1st-4th": "Low",
    "5th-6th": "Low",
    "7th-8th": "Low",
    "9th": "Low",
    "10th": "Low",
    "11th": "Low",
    "12th": "Low",
    "HS-grad": "Medium",
    "Some-college": "Medium",
    "Assoc-acdm": "Medium",
    "Assoc-voc": "Medium",
    "Bachelors": "High",
    "Masters": "High",
    "Doctorate": "High",
    "Prof-school": "High"
}
df["education"] = df["education"].replace(replace_education)

replace_marital = {
    "Married-civ-spouse": "Married",
    "Married-AF-spouse": "Married",
    "Married-spouse-absent": "Married",
    "Never-married": "Single",
    "Widowed": "Other",
    "Divorced": "Other",
    "Separated": "Other"
}
df["marital-status"] = df["marital-status"].replace(replace_marital)

replace_relationship = {
    "Husband": "Spouse",
    "Wife": "Spouse",
    "Own-child": "Child",
    "Other-relative": "Other",
    "Unmarried": "Single",
    "Not-in-family": "Single"
}
df["relationship"] = df["relationship"].replace(replace_relationship)

df["native-country"] = np.where(
    df["native-country"] == "United-States", "United-States", "Non-US"
)

df["income"] = np.where(df["income"] == "<=50K", "Low Income", "High Income")

# =========================
# 3. Split X, y
# =========================
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4. Preprocessing
# =========================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

# =========================
# 5. Pipeline dengan SMOTE + KNN
# =========================
pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=9, weights="distance", p=1))
])

# =========================
# 6. Training
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 7. Evaluasi
# =========================
y_pred = pipeline.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low Income", "High Income"], yticklabels=["Low Income", "High Income"])
plt.title("Confusion Matrix - KNN + SMOTE")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
