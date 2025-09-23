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
    accuracy_score,
    )

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#####################################################################################3
# File Paths
######################################################################################

INPUT_PATH = 
OUTPUT_PATH = 
MODEL_PATH = 



def CancerDetection(Datapath):
    line = "-"*150
    df = pd.read_csv(Datapath)

    return df

def DataModify(df):

    df.drop(columns = "CodeNumber", inplace = True)
    #print(df.head())

    df.replace("?", np.nan, inplace = True)
    print(df.isnull().sum())

    df = df.apply(pd.to_numeric, errors='ignore')

    for col in df.select_dtypes(include = [np.number]).columns:
        df[col].fillna(df[col].median(), inplace = True)

    print(df.isnull().sum())
    
    return df

def Model(df):
    x = df.drop(columns = 'CancerType')
    y = df['CancerType']

    scaler = StandardScaler()

    df = scaler.fit_transform(x)
    df = pd.DataFrame(x)

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)

    print(accuracy*100)
    
    cm = confusion_matrix(y_test, y_predict)


    print(classification_report(y_test, y_predict))

    importances = model.feature_importances_

    # Create DataFrame for better plotting
    feat_df = pd.DataFrame({
        'Feature': x.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title('Feature Importance - Random Forest', fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()
        
def main():
   Dataframe =  CancerDetection("breast-cancer-wisconsin.csv")
   Data = DataModify(Dataframe)
   Model(Data) 

if __name__ == "__main__":
    main()