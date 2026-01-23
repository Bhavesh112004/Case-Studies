import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def MarvellousHeadBrain(datasetpath):
    df = pd.read_csv(datasetpath)

    print("The Data is loaded")

    print("Dimension of the dataset :",df.shape)

    print("Initial data is :")
    print(df.head())

    return df

'''def Visualization(df):

    plt.figure(figsize = (10,8))
    sns.scatterplot(data = df,x = 'Head Size(cm^3)', y = 'Brain Weight(grams)',hue = 'Gender' , palette = 'Set1')
    plt.title("Marvellous HeadBrainData Visualization")
    plt.xlabel("Head Size")
    plt.ylabel("Brain Weight")
    plt.show()'''

def Cleaning(df):

    cleaned_data = df.drop(columns = ['Gender','Age Range'])
    print("The dimesions of the data are : ",cleaned_data.shape)

    print("The cleaned data is : ")
    print(cleaned_data.head())

    return cleaned_data

def Predict(cleaned_data):

    X = cleaned_data['Head Size(cm^3)']
    #print(X.head())
    Y = cleaned_data['Brain Weight(grams)']
    

    # Finding the mean
    X_sum = 0
    Y_sum = 0
    for i in range(len(X)):
        X_sum = X_sum + X[i]
        Y_sum = Y_sum + Y[i]

    mean_X = X_sum / len(X)
    mean_Y = Y_sum / len(Y)

    print("Mean of Independent Variable is : ",mean_X)
    print("Mean of Dependent Variable is : ",mean_Y)

    #Y = mX + c
    # Finding the slope
    n = len(X)

    numerator = 0
    denominator = 0
    
    for i in range (n):
        numerator = numerator + (X[i]-mean_X) * (Y[i]- mean_Y)
        denominator = denominator + (X[i] - mean_X)**2

    m = numerator / denominator

    print("Slop is : ",m)

    # Finding the slope

    # from y = mx + c
    # c = y - mx

    C = mean_Y -(m * mean_X)
    print("Y intercept is : ",C)

    # Getting the R**2 value

    numerator1 = 0
    denominator1 = 0
    for i in range(len(Y)):
        y_predict = m *X[i] + C
        numerator1 = numerator1 + (Y[i]- y_predict) **2
        denominator1 = denominator1 + (Y[i] - mean_Y) **2

    r_square = 1 - (numerator1 / denominator1)

    print("The value of R Square is : ",r_square)
    

def main():

    Data = MarvellousHeadBrain("MarvellousHeadBrain.csv")
    #Visual = Visualization(Data)
    cleaned_data = Cleaning(Data)
    prediction = Predict(cleaned_data)

if __name__ == "__main__":
    main()
