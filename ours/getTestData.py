import numpy as np
import matplotlib.pyplot as plt


def getData():
    k, b = [6, 2, -3, -0.8], [10, -9, 8, -7]
    x = np.array([-3 + 0.1*i for i in range(61)])
    result = []
    for i in range(4):
        y = k[i] * x + b[i] + 4. * (np.random.rand(len(x)) - 0.5)
        result.append(np.vstack((x, y)))
    data = np.hstack((result[0], result[1], result[2], result[3]))
    return data
    # testData = data.T

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = getData()
    x = data[0]
    y = data[1]
    ax.scatter(x, y)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.savefig('ellipsoid.png')