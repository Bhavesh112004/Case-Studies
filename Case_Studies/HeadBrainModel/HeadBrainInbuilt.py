import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def MarvellousLinearRegression(datasetpath):
    df = pd.read_csv(datasetpath)

    print("Dimension of the dataset :", df.shape)
    print("Initial data is :")
    print(df.head())

    # Optional Visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Head Size(cm^3)', y='Brain Weight(grams)', hue='Gender', palette='Set1')
    plt.title("Marvellous Head Size vs Brain Weight")
    plt.xlabel("Head Size (cm³)")
    plt.ylabel("Brain Weight (grams)")
    plt.grid(True)
    plt.show()

    # Cleaning unnecessary columns
    df = df.drop(columns=['Gender', 'Age Range'])

    X = df[['Head Size(cm^3)']]
    Y = df['Brain Weight(grams)']

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Predict on test set
    Y_pred = model.predict(X_test)

    # R² score
    r2 = r2_score(Y_test, Y_pred)
    print("R-squared on test data is:", r2)

    # Show regression line on test set
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, Y_test, color='red', label='Actual')
    plt.plot(X_test, Y_pred, color='blue', label='Predicted Line')
    plt.title("Linear Regression on Test Data")
    plt.xlabel("Head Size (cm³)")
    plt.ylabel("Brain Weight (grams)")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    MarvellousLinearRegression("MarvellousHeadBrain.csv")

if __name__ == "__main__":
    main()
