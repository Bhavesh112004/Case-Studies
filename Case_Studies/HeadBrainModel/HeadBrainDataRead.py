import pandas as pd

def MarvellousHeadBrain(datasetpath):
    df = pd.read_csv(datasetpath)

    print("The Data is loaded")

    print("Dimension of the dataset :",df.shape)

    print("Initial data is :")
    print(df.head())


def main():

    Data = MarvellousHeadBrain("MarvellousHeadBrain.csv")

if __name__ == "__main__":
    main()
