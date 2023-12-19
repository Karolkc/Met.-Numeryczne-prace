import numpy as np
import matplotlib.pyplot as plt

#ilość neuonów w warstwie ukrytej, stąd też kształt macierzy dla wag i biasów
hidden_neurons = 10

weights_hidden = np.random.rand(hidden_neurons) - 0.5
bias_hidden = np.random.rand(hidden_neurons) - 0.5
weight_output = np.random.rand(hidden_neurons) - 0.5
bias_output = np.random.rand() - 0.5

#aktywacja dla neuronów w warstwie ukrytej
def relu(x):
    return np.maximum(0, x)

def loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def predict(x):
    out = np.zeros((x.shape[0],1))
    k = 0
    for i in x:
        hidden_layer_output = relu(i * weights_hidden + bias_hidden)
        out[k] = np.sum(hidden_layer_output * weight_output) + bias_output
        k += 1

    return out.T


def train(x, y, lr=0.001, epochs=3000):
    global weights_hidden, bias_hidden, weight_output, bias_output
    for epoch in range(epochs):
        for i in range(len(x)):
            #forward prop
            hidden_layer_output = relu(x[i] * weights_hidden + bias_hidden)
            y_pred = np.sum(hidden_layer_output * weight_output) + bias_output

            error = y_pred - y[i]

            #backward prop
            grad_weight_output = hidden_layer_output * error
            grad_bias_output = error
            grad_weights_hidden = x[i] * error * weight_output * (hidden_layer_output > 0) #prosta aktywacja
            grad_bias_hidden = error * weight_output * (hidden_layer_output > 0)  #prosta aktywacja

            weight_output -= lr * grad_weight_output
            bias_output -= lr * grad_bias_output
            weights_hidden -= lr * grad_weights_hidden
            bias_hidden -= lr * grad_bias_hidden
        if epoch % 300 == 0:
            y_preds = np.array(predict(x))
            print(f"Epoch {epoch}: Loss = {loss(y, y_preds)}")

def model_plot(x, y, predict):
    plt.style.use('dark_background')

    x_args = np.linspace(min(x), max(x), 200).reshape(-1, 1)
    y_args = predict(x_args).flatten()
    plt.scatter(x, y, color='blue', label='trening')
    plt.plot(x_args, y_args, color='pink', label='model')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('trening a model')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("kw")
    x_train = np.array([0, 2, 1, 4, 18])
    y_train = np.array([0, 4, 2, 8, 36])

    train(x_train, y_train)

    x_test = np.array([1, 8, 5, 14])

    predictions = predict(x_test)
    print(predictions)

    model_plot(x_train, y_train, predict)
