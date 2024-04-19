from neural.containers import Sequential
import neural.containers
import neural.layers
from neural.losses import MSE
from neural.layers import Dense, TanH, RELU
import numpy as np 
import neural
import neural.losses as loss 
loss.MSE()

# training data
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([0, 1, 1, 0])

# network
net = Sequential([
    Dense(2, 3),
    RELU(),
    Dense(3, 1),
    TanH()])

net.display_network()

print(Dense(2, 3).forward(x_train))
# train
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1, loss_func=MSE())



