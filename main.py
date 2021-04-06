import numpy as np #linear algebra
import pandas as pd # a data processing and CSV I/O library
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import csv
SETOSA="setosa"
VERSICOLOR="versicolor"
VIRGINICA="virginica"
COLORS = ['blue', 'red', 'green']


class Main:

    def main(self):
        sepal_length,sepal_width,petal_length,petal_width,iris=self.get_data()
        self.get_table()
        self.draw(sepal_length,sepal_width,petal_length,petal_width,iris)

    def get_data(self):

        # get the dataset
        iris = datasets.load_iris()

        # To get the same feature from all classes.
        sepal_length = iris.data.T[0]
        sepal_width = iris.data.T[1]
        petal_length = iris.data.T[2]
        petal_width = iris.data.T[3]
        return sepal_length,sepal_width,petal_length,petal_width,iris


    def draw(self,sepal_length,sepal_width,petal_length,petal_width,iris):
        self.histogram(iris)
        self.scatter_2D(sepal_length,sepal_width,"Sepal length in cm","Sepal width in cm")
        self.scatter_2D(petal_length,petal_width,"Petal length in cm","Petal width in cm")
        self.scatter_2D(sepal_length,petal_width,"Sepal length in cm","Petal width in cm")
        self.scatter_2D(petal_length,sepal_width,"Petal length in cm","Sepal width in cm")
        self.scatter_2D(sepal_length,petal_length,"Sepal length in cm","Petal length in cm")
        self.scatter_2D(petal_width,sepal_width,"Petal width in cm","Sepal width in cm")
        self.scatter_3D(sepal_length,sepal_width,petal_length,"Sepal length","Sepal width","Petal length")
        self.scatter_3D(sepal_length,sepal_width,petal_width,"Sepal length","Sepal width","Petal width")
        self.scatter_3D(sepal_width,petal_length,petal_width,"Sepal width","Petal length","Petal width")
        self.scatter_3D(sepal_length,petal_length,petal_width,"Sepal length","Petal length","Petal width")
        self.correlation_matrix(iris)
        var2 = {"Sepal length": sepal_length, "Sepal width": sepal_width,
                "Petal length": petal_length, "Petal width": petal_width}
        for key, value in var2.items():
            self.box_plot(value, key)


    def scatter_2D(self, x,y,x_label,y_label):
        plt.scatter(x[:49],y[:49],c="blue",label=SETOSA)
        plt.scatter(x[50:99],y[50:99],c="red",label=VERSICOLOR)
        plt.scatter(x[100:150],y[100:150],c="green",label=VIRGINICA)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='lower right')

        plt.show()

    def histogram(self,iris):

        fig, axes = plt.subplots(nrows=2, ncols=2)

        for i, ax in enumerate(axes.flat):
            for label, color in zip(range(len(iris.target_names)), COLORS):
                ax.hist(iris.data[iris.target == label, i], label=
                iris.target_names[label], color=color)
                ax.set_xlabel(iris.feature_names[i])
                ax.legend(loc='upper right',prop={"size":6})
        plt.show()

    def scatter_3D(self, x,y,z,x_label,y_label,z_label):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:49],y[:49],z[:49],c="blue",label=SETOSA)
        ax.scatter(x[50:99], y[50:99],z[50:99], c="red", label=VERSICOLOR)
        ax.scatter(x[100:150], y[100:150],z[100:150], c="green", label=VIRGINICA)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.legend(loc='upper right')



        plt.show()

    def correlation_matrix(self,iris):
        corr = np.corrcoef(iris.data.T)
        sns.heatmap(corr,
                    annot=True, cmap='cubehelix_r')
        plt.show()
        #Observation---> The Sepal Width and Length are not correlated The Petal Width and Length are highly correlated.

    def box_plot(self,data,y_label):
        my_dict = {SETOSA: data[:49], VERSICOLOR: data[50:99],
                   VIRGINICA:data[100:149]}
        fig, ax = plt.subplots()
        box=ax.boxplot(my_dict.values(),patch_artist=True)
        ax.set_xticklabels(my_dict.keys())
        ax.set_ylabel(y_label)
        for patch, color in zip(box['boxes'], COLORS):
            patch.set_facecolor(color)
        plt.show()

    def get_table(self):
        reader = csv.reader(open("iris.csv"))
        for row in reader:
            print(row)

m=Main()
m.main()