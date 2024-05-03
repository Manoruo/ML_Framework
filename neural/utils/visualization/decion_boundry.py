import matplotlib.pyplot as plt 
import matplotlib
from . import np

# Define colormaps for each class
colormaps = {
    0: plt.cm.Reds,
    1: plt.cm.Blues,
    2: plt.cm.Greens,
    3: plt.cm.Greys,
    4: plt.cm.Purples,
    5: plt.cm.Oranges
}
colormap2 = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'grey',
    4: 'purple',
    5: 'orange'
}

ADDITIONAL_SPACE  = .2 

def make_grid(X_feature):
    
    # define bounds of the domain
    min1, max1 = X_feature[:, 0].min() - ADDITIONAL_SPACE, X_feature[:, 0].max() + ADDITIONAL_SPACE
    min2, max2 = X_feature[:, 1].min() - ADDITIONAL_SPACE, X_feature[:, 1].max() + ADDITIONAL_SPACE
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)

    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    return grid

def plot_decision_2d(model, X, y_act=None):

    # make mesh around wherever the data is centered 
    grid = make_grid(X)

    # get predictions of the model on this mesh + confidence (assuming model uses softmax)
    y_pred = model.predict(grid)
    y_class = np.argmax(y_pred, axis=1)
    y_confidence = np.max(y_pred, axis=1)
    
    # go through each class predicted on the mesh and create decion boundry for that class 
    for class_id in np.unique(y_class):
        x = grid[y_class == class_id]
        confidence = y_confidence[y_class == class_id]
        mapping = colormaps[class_id % len(colormaps.keys())]
        scatter = plt.scatter(x[:, 0], x[:, 1],  c=confidence, cmap=mapping)
        #plt.colorbar(scatter, pad=0.03, fraction=0.05, ticks=[], label='Class ' + str(class_id))

    # plot the actual data on the graph 
    if y_act is not None:
        y_act_class = np.argmax(y_act, axis=1)
        for class_id in np.unique(y_act_class):
            mapping2 = colormap2[class_id % len(colormaps.keys())]
            x2 = X[y_act_class == class_id] # get the points that are of this class
            plt.scatter(x2[:, 0], x2[:, 1], color=mapping2, label='Class ' + str(class_id), s=32, edgecolor='black')

    plt.title(f'Model Decision Boundary')
    plt.tight_layout()
    plt.legend()
    # Show the plot
    plt.show()