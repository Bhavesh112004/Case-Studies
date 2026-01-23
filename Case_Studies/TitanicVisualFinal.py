import pandas as pd
import numpy as np

from matplotlib.pyplot import figure,show
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import countplot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix

def MarvellousTitanicLogistic(Datapath):
    df = pd.read_csv(Datapath)
    print("Datset loaded successfully")
    print(df.head())

    print("Dimensions of dataset is : ",df.shape)

    df.drop(columns = ['Passengerid','zero'], inplace = True)
    print("Dimensions of dataset is : ",df.shape)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

    figure()
    target = "Survived"
    countplot(data = df, x = target).set_title('MarvellousTitanic')
    show()

    figure()
    target = "Survived"
    countplot(data = df, x = target, hue = "Sex").set_title("Based on Gender")
    show()

    figure()
    target = "Survived"
    countplot(data = df, x = target, hue = "Pclass").set_title("Based on Passenger class")
    show()

    figure()
    df['Age'].plot.hist().set_title("Age Report")
    show()

    figure()
    df['Fare'].plot.hist().set_title("Fare Report")
    show()

    plt.figure(figsize = (10,6))
    sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

    x = df.drop(columns = ['Survived'])
    y = df['Survived']

    print('Dimension of target : ',x.shape)
    print('Dimension of labels : ',y.shape)

    scalar = StandardScaler()
    x_scale = scalar.fit_transform(x)

    x_train,x_test,y_train,y_test = train_test_split(x_scale,y, test_size = 0.2, random_state = 42)

    model = LogisticRegression()
    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)

    accuracy = accuracy_score(y_test,y_predict)
    cm = confusion_matrix(y_test,y_predict)

    print("Accuracy is : ",accuracy)
    print("Confusion matrix : ")
    print(cm)

def main():

    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")

if __name__ == "__main__":
    main()