import numpy as np

#globalne domyślne zmienne
weight = np.random.rand() - 0.5 #dla bliskości parametru do 0
bias = np.random.rand() - 0.5

def loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def predict(x):
    return weight * x + bias

#dla sieci bez warstwy ukrytej nie robię funkcji aktywacji
def train(x, y, lr=0.01, epochs=1000):
    global weight, bias
    for epoch in range(epochs):
        y_pred = predict(x)
        error = y_pred - y
        weight -= lr * np.dot(error, x) / len(x)
        bias -= lr * np.mean(error)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss(y, y_pred)}")
            #print(weight, bias)
        if (epoch == epochs-1):
            print(f"ostateczna postać funkcji liniowej: {weight}x + {bias}")


if __name__ == "__main__":
    print("kw")
    x_train = np.array([0, 2, 1, 4, 18])
    y_train = np.array([0, 4, 2, 8, 36])

    train(x_train, y_train)

    x_test = np.array([1, 8, 5, 14])
    predictions = predict(x_test)
    print(predictions)

