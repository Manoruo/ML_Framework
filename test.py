from neural.containers import Sequential
from neural.losses import CE
from neural.utils.preprocessing import OneHotEncoder
from neural.layers import RELU, Softmax, Dense, TanH, Sigmoid
import datasets.spiral_data as sd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 

num_classes = 5 

(x, y) = sd.generate_spiral_data(n_samples=100, n_class=num_classes, noise=.1)

# encode the y's
encoder = OneHotEncoder()
encoder.fit(y)
y_encoded = encoder.encode(y)

X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=.2)



model = Sequential([
    Dense(2, 5),
    TanH(),
    Dense(5, 8),
    TanH(),
    Dense(8, num_classes),
    Softmax()
])
import time
#model.display_network()
num_epochs = 1000
strt = time.time()
loss, acc, times = model.fit(X_train, y_train, num_epochs, .1, CE(), accuracy="categorical", batch_size=64)
print("Total Time:", time.time() - strt)

fig, axes = plt.subplots(1, 3, figsize=(15, 8))
axes = axes.flatten()


epochs = np.arange(0, num_epochs, 1)


axes[0].plot(times, label='time')
axes[0].set_title("epochs vs time")

axes[1].plot(loss, label='Loss')
axes[1].set_title('epochs vs loss')

axes[2].plot(acc, label='accuracy')
axes[2].set_title('epochs vs accuracy')
plt.show()

y_pred = encoder.decode(model.predict(X_test))
y_act = encoder.decode(y_test)
print(sum(y_pred == y_act) / len(y_test))

