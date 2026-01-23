import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def main():
    df = pd.read_csv("iris.csv")

    X = df.iloc[:,[0,1,2,3]].values

    WCSS = []
    
    for k in range(1,11):
        model = KMeans(n_clusters = k, init= 'k-means++', n_init= 10, random_state = 42)
        model.fit(X)
        print(model.inertia_) #WCSS
        WCSS.append(model.inertia_)

    plt.plot(range(1,11), WCSS, marker = 'o')
    plt.title("Elbow method for KMeans")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS value")
    plt.grid(True)
    plt.show()


    

if __name__ == "__main__":
    main()