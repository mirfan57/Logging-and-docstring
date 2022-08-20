import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def prepare_data(df, target_col="y"):
    X = df.drop(target_col, axis=1)

    y = df[target_col]

    return X, y


def save_plot(df, model, filename="plot.png", plot_dir="plots"):
    def _create_base_plot(df):  # creating a scatter plot of data given to dataframe
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)  # building x-axis or horizontal line
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)  # building y-axis or vertical line

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)

        X = X.values  # to get array values
        x1 = X[:, 0]  # all the rows and 0th column
        x2 = X[:, 1]  # all the rows and 1st col

        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        # create decision boundaries using meshgrid
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               # resolution means at what range you want to restrict graph
                               np.arange(x2_min, x2_max, resolution))

        y_hat = classifier.predict(
            np.array([xx1.ravel(), xx2.ravel()]).T)  # ravel to flatten the matrix into single array
        y_hat = y_hat.reshape(xx1.shape)

        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())  # x-limit
        plt.ylim(xx2.min(), xx2.max())  # y-limit

        plt.plot()  # to plot the graph in same jupyter nb

    X, y = prepare_data(df)

    _create_base_plot(df)
    _plot_decision_regions(X, y, model)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)