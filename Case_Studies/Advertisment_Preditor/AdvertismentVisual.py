import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def MarvellousAdvertise(datapath):
    df = pd.read_csv(datapath)

    print("Dataset sample is ")
    print(df.head())

    print("Clean the dataset : ")
    df.drop(columns = ['Unnamed: 0'], inplace = True)

    print(df.head())

    print("Missing values in each columnn : ",df.isnull().sum())

    print("Statistical Summary : ")
    print(df.describe())

    print("Correlation Matrix ")
    print(df.corr())

    plt.figure(figsize = (10,5))
    sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
    plt.title("Marvellous Correalation Heatmap")
    plt.show()

    sns.pairplot(df)
    plt.suptitle("Pairplot of Features", y= 1.02)
    plt.show()


def main():
    MarvellousAdvertise("Advertising.csv")

if __name__ == "__main__":
    main()