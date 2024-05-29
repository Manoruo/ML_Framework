from neural.containers import Sequential
from neural.losses import CE
from neural.utils.preprocessing import OneHotEncoder
from neural.layers import RELU, Softmax, Dense, TanH, Sigmoid
import datasets.spiral_data as sd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

num_classes = 5 

(x, y) = sd.generate_spiral_data(n_samples=100, n_class=num_classes, noise=.4)

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

#model.display_network()

loss, acc = model.fit(X_train, y_train, 1000, .001, CE(), accuracy="categorical", batch_size=64)
plt.plot(loss, label='Loss')
plt.plot(acc, label='accuracy')
plt.legend()
plt.show()

y_pred = encoder.decode(model.predict(X_test))
y_act = encoder.decode(y_test)
print(sum(y_pred == y_act) / len(y_test))

