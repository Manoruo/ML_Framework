import numpy as np 

def generate_spiral_data(n_samples, n_class=2, noise=0.5):
    """
    Generate a synthetic spiral dataset.

    Parameters:
    - n_samples: Number of samples to generate.
    - n_class: Number of classes (default is 2 for a binary classification problem).
    - noise: Noise level (standard deviation of added Gaussian noise).

    Returns:
    - X: Input features.
    - y: Labels.
    """
    X = np.zeros((n_samples * n_class, 2))
    y = np.zeros(n_samples * n_class, dtype='uint8')
    for i in range(n_class):
        ix = range(n_samples * i, n_samples * (i + 1))
        r = np.linspace(0.0, 1, n_samples)  # radius
        t = np.linspace(i * 2 * np.pi / n_class, (i + 2) * 2 * np.pi / n_class, n_samples) + np.random.randn(n_samples) * noise  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = i
    return X, y

def plot_spiral(ax, x_data, y_data, num_classes, title="Spiral Dataset", correct=None):
    
    # plot points 
    for class_index in range(num_classes):
        ax.scatter(x_data[y_data == class_index, 0], x_data[y_data == class_index, 1], label=f'Class {class_index}')
        
    if (correct) is not None:
        wrong = x_data[y_data != correct]
        ax.scatter(wrong[:, 0], wrong[:, 1], label='Incorrect', marker='x', color='black')
    

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title)
    ax.legend()