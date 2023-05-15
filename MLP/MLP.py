# MLP algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


###########################
# TESTING
###########################

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int_)

mlp = MLPClassifier(max_iter=1, learning_rate_init=0.01, random_state=None, warm_start=True)

# Pre-define the axes for plotting
axes = [0, 7, 0, 3]

# Pre-generate a grid of sampling points
x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 200).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )

results = list()

# Now, show the change after fitting epoch by epoch
for epochs in range(0,100):
    
    # Fit the model
    mlp.fit(X, y)

    # Use to model to sampling predictions over all feature space
    y_predict = mlp.predict(np.c_[x0.ravel(), x1.ravel()])
    zz = y_predict.reshape(x0.shape)
    
    results.append(zz)

# prepare plot and colors
fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (10,8)) 
fig.tight_layout()
fig.subplots_adjust(hspace=0.5)
viridis_big = mpl.colormaps['viridis']
custom_cmap = ListedColormap(viridis_big(np.linspace(0.6, 0.75, 128)))

iterator = 1
for n in range(2):
    for i in range(3):
        # plot the dataset
        axs[i, n].plot(X[y==1, 0], X[y==1, 1], "o", label="Iris-Setosa", c = "#598cbd")
        axs[i, n].plot(X[y==0, 0], X[y==0, 1], "o", label="Not Iris-Setosa", c = "#3e6182")

        # plot contour plot
        color = axs[i, n].contourf(x0, x1, results[iterator], cmap=custom_cmap)
        axs[i, n].set_title('Run for epoch nr: ' + str(iterator), fontweight ="bold", fontsize=10, loc='left')
        axs[i, n].set_xlabel("Petal length", fontsize=10)
        axs[i, n].set_ylabel("Petal width", fontsize=10)
        axs[i, n].axis(axes)
        iterator = iterator + (n+1) * (i+1)

# legend and colorbar
cbar = fig.colorbar(color, orientation = 'horizontal', ax=axs.ravel().tolist(), ticks=[0, 1], aspect = 50, shrink = 0.95)
cbar.ax.set_xticklabels(['not-setosa', 'setosa'])
axs[2,1].legend(loc='lower center', bbox_to_anchor = (-0.1, -0.7) , fancybox=True, shadow = True, ncols = 2)

plt.show()