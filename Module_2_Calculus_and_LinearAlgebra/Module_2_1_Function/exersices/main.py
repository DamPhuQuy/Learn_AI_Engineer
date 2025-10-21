from typing import Literal
import math
import matplotlib.pyplot as plt


def ham_mu(x: float, a: float):
    return a**x


def sigmoid(x: float, a: float):
    return 1 / (1 + float(a) ** (-x))


def gaussian(x: float, mu: float = 0, sigma: float = 1):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def softmax(values: list[float]) -> list[float]:
    exp_values: list[float] = [math.exp(e) for e in values]
    sum_exp_values: float | Literal[0] = sum(exp_values)
    return [x / sum_exp_values for x in exp_values]


def logarithm(x: float, base: float = math.e) -> float:
    return math.log(x, base)


def binary_crossentropy(y, y_hat):
    return -y * math.log(y_hat) - (1 - y) * math.log(1 - y_hat)


if __name__ == "__main__":
    a: float = float(input("Nhập hệ số a: "))

    x = [i for i in range(-20, 21)]
    x_positive = [i for i in range(1, 21)]

    y_exp = [ham_mu(i, a) for i in x]
    y_sig = [sigmoid(i, a) for i in x]
    y_gauss = [gaussian(i) for i in x]
    y_ln = [logarithm(i) for i in x_positive]
    y_log10 = [logarithm(i, 10) for i in x_positive]
    y_log1_2 = [logarithm(i, 1 / 2) for i in x_positive]

    softmax_values = softmax(x)

    y_hat = [i * 0.01 for i in range(1, 100)]
    loss_y1 = [binary_crossentropy(1, y_element) for y_element in y_hat]
    loss_y0 = [binary_crossentropy(0, y_element) for y_element in y_hat]

    plt.figure(figsize=(12, 10))  # tuple(12, 10)

    # ham mu
    plt.subplot(2, 3, 1)
    plt.plot(x, y_exp)
    plt.title("ham mu")
    plt.grid(True)

    # sigmoid
    plt.subplot(2, 3, 2)
    plt.plot(x, y_sig)
    plt.title("ham sigmoid")
    plt.grid(True)

    # gauss
    plt.subplot(2, 3, 3)
    plt.plot(x, y_gauss)
    plt.title("gauss")
    plt.grid(True)

    # logarithm
    plt.subplot(2, 3, 4)
    plt.title("logarihm")
    plt.plot(x_positive, y_ln, color="red", label="lnx")
    plt.plot(x_positive, y_log10, color="green", label="log10")
    plt.plot(x_positive, y_log1_2, color="blue", label="log1/2")
    plt.grid(True)
    plt.legend()

    # softmax
    plt.subplot(2, 3, 5)
    plt.plot(x, softmax_values)
    plt.grid(True)

    # binary cross entropy

    plt.subplot(2, 3, 6)
    plt.plot(x_vals, loss_y1, label="y=1")
    plt.plot(x_vals, loss_y0, label="y=0")
    plt.title("binary cross entropy")
    plt.xlabel("y-hat")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.show()
