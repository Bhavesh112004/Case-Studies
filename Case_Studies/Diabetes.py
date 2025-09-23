###########################################################################################
# Required Packages
###########################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

###########################################################################################
# File Paths
###########################################################################################
INPUT_PATH = "diabetes.csv"                  
MODEL_PATH = "FN_lr_dt_pipeline.joblib"         
SCALER_PATH = "db_scaler.joblib"            

###########################################################################################
# Headers (only used if CSV has no headers)
###########################################################################################
HEADERS = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
    "BMI","DiabetesPedigreeFunction","Age","Outcome"
]

###########################################################################################

###########################################################################################
def read_data(path, has_header=True):
    """ Read the data into pandas dataframe """
    if has_header:
        data = pd.read_csv(path)  # Use existing headers
    else:
        data = pd.read_csv(path, header=None)  # No headers in file
        data.columns = HEADERS
    return data

###########################################################################################
# Data Preprocessing
###########################################################################################
def handel_missing_values_with_imputer(df):
    """ Handle missing values """
    # Fill missing values with median
    cols_with_missing = [
        HEADERS[1],  # Glucose
        HEADERS[2],  # BloodPressure
        HEADERS[3],  # SkinThickness
        HEADERS[4],  # Insulin
        HEADERS[5],  # BMI
    ]

    df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

    return df

def extract_features_labels_after_scaling(df):
    '''Separate features and target '''
    X = df.drop(columns=['Outcome'])
    Y = df['Outcome']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, Y

def split_dataset(dataset, train_percentage, features, labels, random_state = 42):
    '''Split dataset into train/test'''
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels,
        train_size = train_percentage, random_state = random_state, stratify = labels
    )

    return train_x, test_x, train_y, test_y

def dataset_statistic(dataset):
    '''Print Basic Stats '''
    print(dataset.describe(include = 'all'))

def build_pipeline():

    pipe = Pipeline(steps = [
        ("imputer", SimpleImputer(strategy = "median")),
        ("rf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=0.1,            # Regularization strength (smaller = more regularization)
            solver='liblinear', # Works well for small datasets
            penalty='l2',
            class_weight='balanced' # Adjust for imbalance
        ))
    ])
    return pipe

def train_pipeline(pipeline, X_train, Y_train):
    pipeline.fit(X_train, Y_train)
    return pipeline



def main():

    # Load CSV
    dataset = pd.read_csv(INPUT_PATH)

    # Basic Stats
    dataset_statistic(dataset)

    # Handle Missing Values
    dataset = handel_missing_values_with_imputer(dataset)

    # Prepare Features/Target
    features, labels =  extract_features_labels_after_scaling(dataset)

    print(dataset)
    # Split
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.8, features, labels)

    print("Train_x Shape ::", train_x.shape)
    print("Train_y Shape ::", train_y.shape)
    print("Test_x Shape ::", test_x.shape)
    print("Test_y Shape ::", test_y.shape)

    # Build + Pipeline
    pipeline = build_pipeline()
    trained_model = train_pipeline(pipeline, train_x, train_y)
    print("Trained Pipeline :: ", trained_model)

    # predictions
    predictions = trained_model.predict(test_x)

    # Metrics
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y, predictions))
    print("Classification Report:\n", classification_report(test_y, predictions))
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions))

    # Feature Importance 
    plot_feature_importances(trained_model, features, title = "Feature Importance(LR)")

if __name__ == "__main__":
    main()
