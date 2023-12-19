import numpy as np
import matplotlib.pyplot as plt

#ustawiony na "sztywno" stopień wielomianu
polynomial_degree = 3

weights = np.random.randn(polynomial_degree+1, 1) * 0.01

def loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def predict(x):
    #ręczne wpisanie n nowych cech (przeskalowamych x-ów) jako kolumn macierzy
    x_transformed = np.vstack([x**polynomial_degree, x**polynomial_degree-1, x, np.ones_like(x)]).T
    return np.dot(x_transformed, weights)

def train(x, y, lr=0.0001, epochs=100000):
    global weights
    y = y.reshape(-1, 1)
    for epoch in range(epochs):
        y_pred = np.dot(x, weights)
        error = y_pred - y
        gradients = np.dot(x.T, error) / len(x)
        weights -= lr * gradients

        if epoch % 5000 == 0:
            print(f"Epoch {epoch}: Loss = {loss(y, y_pred)}")

def model_plot(x, y, predict):
    plt.style.use('dark_background')

    x_args = np.linspace(min(x), max(x), 200)
    y_args = predict(x_args)
    plt.scatter(x, y, color='blue', label='trening')
    plt.plot(x_args, y_args, color='pink', label='model')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('trening a model')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("kw")
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 8, 27, 64, 125])

    #znowu ręczne skalowanie
    x_train = np.vstack([x**3, x**2, x, np.ones_like(x)]).T

    train(x_train, y)
    print("\nOstatnie wagi:", weights.flatten())

    x_test = np.array([1, 2, 6])
    predictions = predict(x_test)

    print("Predykcje:", predictions.flatten())
    print("Postać funkcji regresji: {:.2f}x^3 + {:.2f}x^2 + {:.2f}x + {:.2f}".format(*weights.flatten()))

    model_plot(x, y, predict)
