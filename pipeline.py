import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print(" Data Loaded Successfully")
    return df
def preprocess_data(df):
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df = df.drop('customerID', axis=1)

    print(" Preprocessing Completed")
    return df
def feature_engineering(df):
    def tenure_group(x):
        if x <= 12:
            return '0-1 Year'
        elif x <= 24:
            return '1-2 Years'
        elif x <= 48:
            return '2-4 Years'
        else:
            return '4+ Years'

    df['tenure_group'] = df['tenure'].apply(tenure_group)

    df = pd.get_dummies(df, drop_first=True)

    print(" Feature Engineering Completed")
    return df

def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)
def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def train_models(X_train, y_train):

    print("\n Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr_model.fit(X_train, y_train)

    print(" Training XGBoost...")
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    print(" Model Training Completed")

    return lr_model, xgb_model


def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    print(f"\n {model_name} Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def run_pipeline(file_path):

    print(" Running churn pipeline with Logistic Regression + XGBoost") 
    print("Feature branch: XGBoost model improvement applied")
    df = load_data(file_path)
    df = preprocess_data(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test = scale_data(X_train, X_test)

    lr_model, xgb_model = train_models(X_train, y_train)

    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

if __name__ == "__main__":
    run_pipeline("Telecom Customer Churn.csv")