import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def main():

    Data = MarvellousHeadBrain("MarvellousHeadBrain.csv")
    #Visual = Visualization(Data)
    cleaned_data = Cleaning(Data)

if __name__ == "__main__":
    main()
