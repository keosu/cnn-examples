import numpy as np


class LinearRegression(object):
    """
    multivariate linear regression using gradient descent
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        :param learning_rate: learning rate constant
        :param iterations: how many epochs
        """
        self.learning_rate = learning_rate
        self.iterations = iterations

    def gen_data(self, w, b, len=10):
        data = []

        noise = np.random.randn(1) * 0.01
        x0 = np.random.randn(len, w.shape[0])
        y0 = np.dot(x0, w) + b + noise

        return (x0, y0)

    def backprop(self, data, w, b):
        for i in range(data[0].shape[0]):
            x0 = data[0][i]
            y0 = data[1][i]
            y = np.dot(x0, w.T)+b
            w_gradient = (y - y0) * x0.T  # 求w的梯度
            b_gradient = (y-y0)[0]  # 求b的梯度；取标量
            w -= w_gradient*self.learning_rate  # 权值更新
            b -= b_gradient
            loss = 0.5*np.square(y-y0)
        return [w, b, loss]

    def train(self, data):
        w0 = np.random.randn(data[0].shape[1])
        b0 = np.random.randn(1)
        for i in range(self.iterations):
            w0, b0, loss = self.backprop(data, w0, b0)
            if(i % 100 == 0):
                print("iter %3d : loss = %8f" % (i, loss))

        return (w0, b0)

    def predict(self, data, w, b):
        return np.dot(data[0], w) + b


if __name__ == "__main__":
    # y=2*x1+3*x2+1
    w = np.array([2, 3, 4, 5])
    b = np.array([1])

    mode = LinearRegression()

    print("Generating Data...")
    train_data = mode.gen_data(w, b, len=10)
    test_date = mode.gen_data(w, b, len=5)

    print("Train the model...")
    w0, b0 = mode.train(train_data)

    print("Test the model...")
    predict = mode.predict(test_date, w0, b0)

    print("prediction bias:", test_date[1] - predict)
